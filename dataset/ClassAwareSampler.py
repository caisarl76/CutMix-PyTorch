"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import random
import numpy as np
from torch.utils.data.sampler import Sampler
import pdb


##################################
## Class-aware sampling, partly implemented by frombeijingwithlove
##################################

class RandomCycleIter:

    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.i += 1

        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)

        return self.data_list[self.i]


def class_aware_sample_generator(cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0
    while i < n:

        #         yield next(data_iter_list[next(cls_iter)])

        if j >= num_samples_cls:
            j = 0

        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]] * num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]

        i += 1
        j += 1


class ClassAwareSampler(Sampler):

    def __init__(self, data_source, num_samples_cls=1, ):
        num_classes = len(np.unique(data_source.labels))
        self.class_iter = RandomCycleIter(range(num_classes))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(data_source.labels):
            cls_data_list[label].append(i)
        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    def __iter__(self):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    def __len__(self):
        return self.num_samples


def get_sampler():
    return ClassAwareSampler


##################################


if __name__ == '__main__':
    sampler_dic = {
        'sampler': get_sampler(),
        'params': {'num_samples_cls': 4}
    }

    from imbalance_cifar import IMBALANCECIFAR100

    trainset = IMBALANCECIFAR100(phase='train', imbalance_ratio=0.1, root='./data')
    print(len(trainset))

    import torch
    import torch.utils.data

    loader = torch.utils.data.DataLoader(trainset, batch_size=40, num_workers=0, pin_memory=True,
                                         sampler=sampler_dic['sampler'](trainset, **sampler_dic['params']))

    for i, (inputs, labels) in enumerate(loader):
        print(inputs.shape)
        print(labels)
        break
