import numpy as np

def permute_color_channel(x, delta, channel_index, invalid_value_correction='mod'):
    """Uniformly permutes the color channel of an image.

    :param x: Tensor of the image to perturb.
    :param delta: The perturbation value.
    :param channel_index: Index of the color channel to permute.
    :param invalid_value_correction: The method for handling values outside the allowed range. Allowed values are
        'mod' for modulus or 'clip' for clipping.

    :return: Tensor of perturbed image.
    """
    batch_size = len(x)

    for i in range(batch_size):
        new_val = x[i, :, :, channel_index] + delta[i]

        if invalid_value_correction == 'mod':
            new_val = new_val % 1
        elif invalid_value_correction == 'clip':
            new_val = np.clip(new_val, 0, 1)

        x[i, :, :, channel_index] = new_val

    return x
