'''
Descripttion: 
version: 
Author: Luckie
Date: 2021-03-29 02:44:26
LastEditors: Luckie
LastEditTime: 2021-07-27 16:18:31
'''
import random

import torch


class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    # A base class for iterators
    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ClassRandomSampler(Sampler):
    """Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.
    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False, If True, it means that repeated sampling is allowed
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,),
                                      dtype=torch.int64).tolist())

        part1 = [m for m in range(0, 19)]
        random.shuffle(part1)
        part2 = [m for m in range(19, 45)]
        random.shuffle(part2)
        part3 = [m for m in range(45, 225)]
        random.shuffle(part3)
        part4 = [m for m in range(225, 359)]
        random.shuffle(part4)
        part5 = [m for m in range(359, 442)]
        random.shuffle(part5)
        # part4 = [m for m in range(3,80,4)]
        # random.shuffle(part4)
        final = part1 + part2 + random.sample(part3, 20) + random.sample(
            part4, 20) + random.sample(part5, 20)
        return iter(final)

    def __len__(self):
        return len(self.data_source)


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    # Batch sampling
    def __init__(self, sampler, batch_size=2, drop_last=True):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            # if idx == 19 or idx == 39 or idx == 59 or idx == 79:
            #     batch.append(idx-1)
            # else:
            #     batch.append(idx+1)
            if len(batch) == self.batch_size:
                # print("batch index: ", batch)
                # print("length sampler", len(self.sampler)) # 80
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
