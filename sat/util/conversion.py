import numpy as np
import torch
from skimage import color


def nchw_to_nhwc(data):
    """
        Converts numpy data format (batch, channels, width, height)
        to pytorch data format (batch, width, height, channels)
    """
    if is_batched(data):
        return np.transpose(data, (0, 2, 3, 1))
    else:
        return np.transpose(data, (1, 2, 0))


def nhwc_to_nchw(data):
    """
        Converts pytorch data format (batch, width, height, channels)
        to numpy data format (batch, channels, width, height)
    """
    if is_batched(data):
        return np.transpose(data, (0, 3, 1, 2))
    else:
        return np.transpose(data, (2, 0, 1))


def rgb_bgr(image):
    """
        Converts rgb color channels to bgr color channels and vice versa.
    """
    if image.ndim == 4:
        dim = 1
    else:
        dim = 0

    if type(image) == torch.Tensor:
        return torch.flip(image, dims=[dim])
    else:
        return np.flip(image, axis=dim)


def rgb_lab(image, to='lab'):
    """
        Converts images in rgb color space to lab color space and vice versa.
    """
    assert to == 'lab' or to == 'rgb', "Argument 'to' must be either 'lab' or 'rgb'"
    assert type(image) == torch.Tensor, "Image must be of type Tensor"

    if image.is_cuda:
        converted_image = image.detach().cpu()
    else:
        converted_image = image.clone()

    converted_image = nchw_to_nhwc(converted_image)

    if to == 'lab':
        converted_image = color.rgb2lab(converted_image)
    elif to == 'rgb':
        converted_image = color.lab2rgb(converted_image)

    converted_image = torch.Tensor(converted_image)

    return converted_image


def is_batched(data):
    """
        Checks if input tensor is a single image or a batch.
    """
    return len(data.shape) == 4


def unnormalize(tensor, mean, std):
    """
        Reverts image normalization transformation using mean and std.
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
