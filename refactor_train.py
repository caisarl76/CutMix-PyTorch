import argparse
import os
import shutil
import time

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import warnings

from utils.utils import *
from utils import resnet as RN, pyramidnet as PYRM
from utils.losses import *
from dataset.imbalance_cifar import IMBALANCECIFAR100, IMBALANCECIFAR10

warnings.filterwarnings("ignore")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Cutmix PyTorch CIFAR-10, CIFAR-100 and ImageNet-1k Training')
parser.add_argument('--net_type', default='resnet', type=str,
                    help='networktype: resnet, and pyamidnet')
parser.add_argument('--depth', default=32, type=int,
                    help='depth of the network (default: 32)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--lr', '--learning-rate', default=0.25, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--data_root', default='./data', type=str, )
parser.add_argument('--dataset', dest='dataset', default='cifar100_lt', type=str,
                    help='dataset (options: cifar10, cifar100, cifar100_lt, and imagenet)')
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--imb_factor', default=0.1, type=float, help='imbalance factor')
parser.add_argument('--sample_method', default='class_balanced', type=str,
                    choices=['random', 'effective_num', 'inverse_class_freq', 'class_balanced'])
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                    help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
                    help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0.5, type=float, help='cutmix probability')
parser.add_argument('--resize_mix', action='store_true')

parser.set_defaults(bottleneck=True)
parser.set_defaults(verbose=True)

best_err1 = 100
best_err5 = 100


def main():
    global args, best_err1, best_err5
    args = parser.parse_args()
    expname = '_'.join(
        [args.dataset, args.imb_type, (str)(args.imb_factor), args.net_type, (str)(args.depth), args.sample_method,
         (str)(args.beta), (str)(args.cutmix_prob), args.loss_type, ('lr' + (str)(args.lr)), ('bs'+(str)(args.batch_size))])

    args.expname = os.path.join('runs', 'refactor', expname)
    # return
    if not os.path.exists(args.expname):
        os.makedirs(args.expname)

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = IMBALANCECIFAR100(phase='train', imbalance_ratio=args.imb_factor, root=args.data_root,
                                      imb_type=args.imb_type)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(args.data_root, train=False, transform=transform_test),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    cls_num_list = train_dataset.get_cls_num_list()
    print(cls_num_list)
    numberofclass = 100

    print("=> creating model '{}'".format(args.net_type))

    model = RN.ResNet('cifar100_lt', 32, 100, True)  # for ResNet

    model = model.cuda()
    print(model)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.sample_method == 'effective_num':
        # calculate effective number of samples for longtail dataset sampler
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, cls_num_list)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * 100
    elif args.sample_method == 'class_balanced':
        weights = np.ones(len(cls_num_list))
        weights = weights / np.sum(weights) * 100
    else:
        weights = None

    # define loss function (criterion) and optimizer

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    cudnn.benchmark = True

    # log for training
    log_test = open(os.path.join(args.expname, 'log_test.csv'), 'w')
    with open(os.path.join(args.expname, 'args.txt'), 'w') as f:
        f.write(str(args))

    for epoch in range(0, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, weights)

        # evaluate on validation set
        err1, err5, val_loss = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = err1 <= best_err1
        best_err1 = min(err1, best_err1)
        if is_best:
            best_err5 = err5

        print('Current best accuracy (top-1 and 5 error):', best_err1, best_err5)
        save_checkpoint({
            'epoch': epoch,
            'arch': args.net_type,
            'state_dict': model.state_dict(),
            'best_err1': best_err1,
            'best_err5': best_err5,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        log_test.write('%d Epoch err1: %.2f, err5: %.2f \n' % (epoch, err1, err5))
        log_test.flush()
    print('Best accuracy (top-1 and 5 error):', best_err1, best_err5)
    log_test.write('Best accuracy (top-1 and 5 error): %.2f, %.2f' % (best_err1, best_err5))
    log_test.flush()


def train(train_loader, model, criterion, optimizer, epoch, weights=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if args.resize_mix and r < args.cutmix_prob:
            if not args.sample_method == 'random':
                from numpy.random import choice
                # generate mixed samples with effective num.
                prob_dist = [weights[i - 1] for i in target]
                prob_dist = prob_dist / np.sum(prob_dist)
                batch_size = input.size()[0]
                rand_index = choice(np.array(range(batch_size)), batch_size, p=prob_dist)
            else:
                rand_index = torch.randperm(input.size()[0]).cuda()

            lam = np.random.beta(args.beta, args.beta)
            target_a = target
            target_b = target[rand_index]
            resized, bbx1, bby1, bbx2, bby2 = resizemix(input, lam)
            if bbx1*bbx2 == 0:
                output = model(input)
                loss = criterion(output, target)
            else:
                input[:, :, bbx1:bbx2, bby1:bby2] = resized[rand_index, :, :]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                # compute output
                output = model(input)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Train Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))

    return losses.avg

def resizemix(images, lam):
    import torch.nn.functional as F
    size = images.size()

    W = size[2]
    H = size[3]

    res_len = np.sqrt(1. - lam)
    cut_w = np.int(W * res_len)
    cut_h = np.int(H * res_len)
    res_len = np.int(W * res_len)

    if res_len == 0:
        return images, 0, 0, 0, 0


    # uniform
    bbx1 = np.random.randint(W - res_len)
    bby1 = np.random.randint(H - res_len)

    # bbx2 = np.clip(cx + (cut_w // 2), 0, W)
    # bby2 = np.clip(cy + (cut_h // 2), 0, H)
    # print(res_len)
    resized = F.interpolate(images, size=res_len)

    return resized, bbx1, bby1, bbx1 + res_len, bby1 + res_len


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1, err5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))
        top5.update(err5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.verbose == True:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})\t'
                  'Top 5-err {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, args.epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}  Top 5-err {top5.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, args.epochs, top1=top1, top5=top5, loss=losses))
    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = args.expname
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.expname, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    main()
