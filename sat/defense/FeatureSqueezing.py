import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
import math


class FeatureSqueezing:

    def __init__(self, image, bit_depth=7, kernel_size=3):

        assert 8 >= bit_depth > 0
        assert kernel_size > 0

        self.image = image
        self.bit_value = (2 ** bit_depth) - 1
        self.kernel_size = kernel_size
        self.padding = _quadruple(math.floor(kernel_size/2))
        #self.padding = (self.padding - 1, self.padding, self.padding, self.padding - 1) if kernel_size % 2 == 0 else _quadruple(self.padding)

        squeezed = torch.floor(self.image * self.bit_value) / self.bit_value
        squeezed = F.pad(squeezed, self.padding, mode='reflect')
        squeezed = squeezed.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        squeezed = squeezed.contiguous().view(squeezed.size()[:4] + (-1,)).median(dim=-1)[0]

        self.squeezed = squeezed
