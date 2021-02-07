import numpy as np


def permute_color_channel(x, delta, channel_index, invalid_value_correction='mod'):
    batch_size = len(x)

    for i in range(batch_size):
        new_val = x[i, :, :, channel_index] + delta

        if invalid_value_correction == 'mod':
            new_val = new_val % 1
        elif invalid_value_correction == 'clip':
            new_val = np.clip(new_val, 0, 1)

        x[i, :, :, channel_index] = new_val

    return x
