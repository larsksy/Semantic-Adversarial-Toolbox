import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
from sat.defence.Defence import ProcessingDefence
import math


class FeatureSqueezing(ProcessingDefence):
    """
    Implementation of Feature Squeezing defence.

    Original paper: https://arxiv.org/pdf/1704.01155.pdf
    """

    def __init__(self, bit_depth=7, kernel_size=3):
        """


        :param bit_depth: Number of bits used to encode image colors. Use 8 for no bit reduction.
        :param kernel_size: Size of filter to use for blurring. Larger size means more blurring.
            Use kernel size 1 for no blurring.
        """
        super(FeatureSqueezing).__init__()

        assert 8 >= bit_depth > 0
        assert kernel_size > 0

        self.bit_value = (2 ** bit_depth) - 1
        self.kernel_size = kernel_size
        self.padding = _quadruple(math.floor(kernel_size/2))
        # self.padding = (self.padding - 1, self.padding, self.padding, self.padding - 1)
        # if kernel_size % 2 == 0 else _quadruple(self.padding)

    def __call__(self, image):
        """

        :param image: The image to apply the defence to.
        :return: The feature squeezed image
        """
        squeezed = torch.floor(image * self.bit_value) / self.bit_value
        squeezed = F.pad(squeezed, self.padding, mode='reflect')
        squeezed = squeezed.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        squeezed = squeezed.contiguous().view(squeezed.size()[:4] + (-1,)).median(dim=-1)[0]

        return squeezed
