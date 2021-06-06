import torch

"""
Util functions for EdgeFool Attack

https://github.com/smartcameras/EdgeFool

"""

def preprocess_lab(lab):
    """Converts LAB color channels to [-1, 1] data range"""
    L_chan, a_chan, b_chan = torch.unbind(lab, dim=2)
    return [L_chan / 50.0 - 1.0, a_chan / 110.0, b_chan / 110.0]


def deprocess_lab(L_chan, a_chan, b_chan):
    """Converts LAB color channels to original data range"""
    return torch.stack([(L_chan + 1) / 2.0 * 100.0, a_chan * 110.0, b_chan * 110.0], dim=2)


def rgb_to_lab(srgb):
    """Converts rgb to lab color in a differential manner."""
    device = 'cuda' if srgb.is_cuda else 'cpu'
    srgb_pixels = torch.reshape(srgb, [-1, 3])

    linear_mask = (srgb_pixels <= 0.04045).type(torch.FloatTensor).to(device)
    exponential_mask = (srgb_pixels > 0.04045).type(torch.FloatTensor).to(device)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask

    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ]).type(torch.FloatTensor).to(device)

    xyz_pixels = torch.mm(rgb_pixels, rgb_to_xyz)

    # XYZ to Lab
    xyz_normalized_pixels = torch.mul(xyz_pixels,
                                      torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).type(torch.FloatTensor).to(device))

    epsilon = 6.0 / 29.0

    linear_mask = (xyz_normalized_pixels <= (epsilon ** 3)).type(torch.FloatTensor).to(device)

    exponential_mask = (xyz_normalized_pixels > (epsilon ** 3)).type(torch.FloatTensor).to(device)

    fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon ** 2) + 4.0 / 29.0) * linear_mask + (
                (xyz_normalized_pixels + 0.000001) ** (1.0 / 3.0)) * exponential_mask

    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ]).type(torch.FloatTensor).to(device)
    lab_pixels = torch.mm(fxfyfz_pixels, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device)
    return torch.reshape(lab_pixels, srgb.shape)


def lab_to_rgb(lab):
    """Converts lab to rgb color in a differential manner."""

    device = 'cuda' if lab.is_cuda else 'cpu'
    lab_pixels = torch.reshape(lab, [-1, 3])

    # convert to fxfyfz
    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ]).type(torch.FloatTensor).to(device)
    fxfyfz_pixels = torch.mm(lab_pixels + torch.tensor([16.0, 0.0, 0.0]).type(torch.FloatTensor).to(device), lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6.0 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).type(torch.FloatTensor).to(device)
    exponential_mask = (fxfyfz_pixels > epsilon).type(torch.FloatTensor).to(device)

    xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + (
                (fxfyfz_pixels + 0.000001) ** 3) * exponential_mask

    # denormalize for D65 white point
    xyz_pixels = torch.mul(xyz_pixels, torch.tensor([0.950456, 1.0, 1.088754]).type(torch.FloatTensor).to(device))

    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ]).type(torch.FloatTensor).to(device)

    rgb_pixels = torch.mm(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    # clip
    rgb_pixels[rgb_pixels > 1] = 1
    rgb_pixels[rgb_pixels < 0] = 0

    linear_mask = (rgb_pixels <= 0.0031308).type(torch.FloatTensor).to(device)
    exponential_mask = (rgb_pixels > 0.0031308).type(torch.FloatTensor).to(device)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + (
                ((rgb_pixels + 0.000001) ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask

    return torch.reshape(srgb_pixels, lab.shape)


def detail_enhance_lab(img, smooth_img):
    """Enhances edges of image using another smoothened image as comparison"""

    val0 = 15
    val2 = 1
    exposure = 1.0
    saturation = 1.0

    # convert 1,C,W,H --> W,H,C
    img = img.squeeze().permute(1, 2, 0)  # (2,1,0)
    smooth_img = smooth_img.squeeze().permute(1, 2, 0)

    # Convert image and smooth_img from rgb to lab
    img_lab = rgb_to_lab(img)
    smooth_img_lab = rgb_to_lab(smooth_img)

    # do the enhancement
    img_l, img_a, img_b = torch.unbind(img_lab, dim=2)
    smooth_l, smooth_a, smooth_b = torch.unbind(smooth_img_lab, dim=2)
    diff = sig((img_l - smooth_l) / 100.0, val0) * 100.0
    base = (sig((exposure * smooth_l - 56.0) / 100.0, val2) * 100.0) + 56.0
    res = base + diff
    img_l = res
    img_a = img_a * saturation
    img_b = img_b * saturation
    img_lab = torch.stack([img_l, img_a, img_b], dim=2)

    L_chan, a_chan, b_chan = preprocess_lab(img_lab)
    img_lab = deprocess_lab(L_chan, a_chan, b_chan)
    img_final = lab_to_rgb(img_lab)

    return img_final


def sig(x, a):
    """
    Applies a sigmoid function on data in [0-1] range. Then rescales
    the result so 0.5 will be mapped to itself.
    """

    # Apply Sigmoid
    y = 1. / (1 + torch.exp(-a * x)) - 0.5

    # Re-scale
    y05 = 1. / (1 + torch.exp(-torch.tensor(a * 0.5, dtype=torch.float32))) - 0.5
    y = y * (0.5 / y05)

    return y