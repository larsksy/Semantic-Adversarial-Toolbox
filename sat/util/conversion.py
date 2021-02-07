import numpy as np


def nchw_to_nhwc(data):
    if is_batched(data):
        return np.transpose(data, (0, 2, 3, 1))
    else:
        return np.transpose(data, (1, 2, 0))


def nhwc_to_nchw(data):
    if is_batched(data):
        return np.transpose(data, (0, 3, 1, 2))
    else:
        return np.transpose(data, (2, 0, 1))


def is_batched(data):
    return len(data.shape) == 4


def unnormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
