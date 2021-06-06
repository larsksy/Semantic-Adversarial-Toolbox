from sat.attacks.attack import Attack
from sat.models.ModelBuilder import ResnetDilated, PPMDeepsup, weights_init
from sat.models.Segmentation import SegmentationModule
from sat.util.visualization import single_image
from sat.util.conversion import rgb_bgr, rgb_lab, nhwc_to_nchw
from sat.util.tqdm import get_tqdm
from torch import nn
from torchvision import transforms
from skimage import color
import torch
import cv2
import numpy as np
import os


class ColorFool(Attack):
    """Implementation of the ColorFool attack

    Original paper: ColorFool: Semantic Adversarial Colorization
    by Ali Shahin Shamsabadi, Ricardo Sanchez-Matilla and Andrea Cavallaro

    https://github.com/smartcameras/ColorFool
    """

    def __init__(self, classifier, device='cpu'):
        """

        :param classifier: The model to fool.
        :param device: 'cuda' if running on gpu, 'cpu' otherwise.
        """
        super(ColorFool, self).__init__(classifier, device)

        # Setup segmentation module
        net_encode = ResnetDilated(dilate_scale=8)
        path_encode = os.path.join(os.getcwd(), 'pretrained/colorfool_segmentation_encoder.pth')
        net_encode.load_state_dict(
            torch.load(path_encode, map_location=lambda storage, loc: storage), strict=False)

        net_decode = PPMDeepsup(fc_dim=2048, num_class=150, use_softmax=True)
        path_decode = os.path.join(os.getcwd(), 'pretrained/colorfool_segmentation_decoder.pth')

        net_decode.apply(weights_init)
        net_decode.load_state_dict(
            torch.load(path_decode, map_location=lambda storage, loc: storage), strict=False)

        crit = nn.NLLLoss(ignore_index=-1)

        self.segmentation = SegmentationModule(net_encode, net_decode, crit)

    def __call__(self, images, labels, max_iterations=1000, generate_samples=False):
        """Performs the attack

        :param images: Images to perturb.
        :param labels: Labels of original images.
        :param max_iterations: Maximum number of trials to find an adversarial image.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.

        :return: List of adversarial images
        """
        water_masks, sky_masks, grass_masks, person_masks = self.segment(images)

        for i in range(images.shape[0]):

            image = images[i]
            image_lab = rgb_lab(image, to='lab')
            water_mask = water_masks[i]
            grass_mask = grass_masks[i]
            sky_mask = sky_masks[i]
            person_mask = person_masks[i]
            label = labels[i].item()

            # Start iteration
            for trial in range(max_iterations):

                X_lab = image_lab.clone().numpy()

                margin = 127
                mult = float(trial + 1) / float(max_iterations)

                a_blue, b_blue = self._perturb_with_mask(water_mask, margin, mult, X_lab, mask_type='water')
                a_green, b_green = self._perturb_with_mask(grass_mask, margin, mult, X_lab, mask_type='grass')
                a_sky, b_sky = self._perturb_with_mask(sky_mask, margin, mult, X_lab, mask_type='sky')

                mask = (person_mask + water_mask + grass_mask + sky_mask)
                mask[mask > 1] = 1

                # Smooth boundaries between sensitive regions
                mask = cv2.blur(mask, (10, 10))

                # Adversarial color perturbation for non-sensitive regions
                random_mask = 1 - mask
                a_random = np.full((X_lab.shape[0], X_lab.shape[1]),
                                   np.random.uniform(mult * (-margin), mult * (margin), size=(1)))
                b_random = np.full((X_lab.shape[0], X_lab.shape[1]),
                                   np.random.uniform(mult * (-margin), mult * (margin), size=(1)))
                a_random_mask = a_random * random_mask
                b_random_mask = b_random * random_mask

                # Adversarialy perturb color (i.e. a and b channels in the Lab color space) of the clean image
                noise_mask = np.zeros((X_lab.shape), dtype=float)
                noise_mask[:, :, 1] = a_blue + a_green + a_sky + a_random_mask
                noise_mask[:, :, 2] = b_blue + b_green + b_sky + b_random_mask
                X_lab_mask = np.zeros((X_lab.shape), dtype=float)
                X_lab_mask[:, :, 0] = X_lab[:, :, 0]
                X_lab_mask[:, :, 1] = np.clip(X_lab[:, :, 1] + noise_mask[:, :, 1], -margin, margin)
                X_lab_mask[:, :, 2] = np.clip(X_lab[:, :, 2] + noise_mask[:, :, 2], -margin, margin)

                # Transfer from LAB to RGB
                X_rgb_mask = color.lab2rgb(X_lab_mask)
                X_rgb_mask = torch.FloatTensor(nhwc_to_nchw(X_rgb_mask))
                X_rgb_mask = X_rgb_mask.to(self.device).unsqueeze(dim=0)

                if self.verify_and_add_adversarial(X_rgb_mask, image, label, generate_samples=generate_samples):
                    single_image(X_rgb_mask, data_format='nchw')
                    break

            single_image(X_rgb_mask, data_format='nchw')

            return self.list

    def _perturb_with_mask(self, mask, margin, mult, X_lab, mask_type='water'):
        """Perturbs segments of an image using image masks.

        :param mask: The segmentation mask.
        :param margin: The perturbation margin.
        :param mult: Controls the intensity of the perturbation.
        :param X_lab: The image to perturb encoded in lab color space.
        :param mask_type: Which segment of the image should be perturbed.

        :return: Perturbed a and b channels of image.
        """

        mask_binary = mask.copy()
        mask_binary[mask_binary > 0] = 1
        mask_result = X_lab[mask_binary == 1]
        if mask_result.size != 0:
            a_min = mask_result[:, 1].min()
            a_max = np.clip(mask_result[:, 1].max(), a_min=None, a_max=0)
            b_min = mask_result[:, 2].min()
            b_max = np.clip(mask_result[:, 2].max(),
                            a_min=0 if mask_type == 'grass' else None,
                            a_max=None if mask_type == 'grass' else 0)

            if mask_type == 'grass':
                a_unif = np.random.uniform(mult * (-margin - a_min), mult * (-a_max), size=(1))
                b_unif = np.random.uniform(mult * (-b_min), mult * (margin - b_max), size=(1))
            else:
                a_unif = np.random.uniform(mult * (-margin - a_min), mult * (-a_max), size=(1))
                b_unif = np.random.uniform(mult * (-margin - b_min), mult * (-b_max), size=(1))

            a = np.full((X_lab.shape[0], X_lab.shape[1]), a_unif) * mask
            b = np.full((X_lab.shape[0], X_lab.shape[1]), b_unif) * mask
        else:
            a = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)
            b = np.full((X_lab.shape[0], X_lab.shape[1]), 0.)

        return a, b

    def segment(self, images, segment_sizes=None, smooth_mask=True):
        """Segments images by semantic regions.

        :param images: Images to segment.
        :param segment_sizes: List of sizes to use to find segments.
        :param smooth_mask: Smoothens the classification values by probability.

        :return: Masks of different image segments.
        """
        self.segmentation.eval()

        if segment_sizes is None:
            segment_sizes = [300, 400, 500, 600]

        water_masks = list()
        sky_masks = list()
        grass_masks = list()
        person_masks = list()

        pbar = get_tqdm('Segmentation', len(images))

        for image in images:

            segSize = (image.shape[1],
                       image.shape[2])

            img_resized_list = get_resize_list(image, segment_sizes)

            with torch.no_grad():
                scores = torch.zeros(1, 150, segSize[0], segSize[1])
                for img in img_resized_list:
                    # forward pass
                    pred_tmp = self.segmentation(img, segSize=segSize)
                    scores += (pred_tmp.cpu() / len(segment_sizes))

                pred_prob, pred = torch.max(scores, dim=1)
                pred = pred.squeeze(0).cpu().numpy()
                pred_prob = pred_prob.squeeze(0).cpu().numpy()

            # visualization
            single_image(pred, save=True, file_name='colorfool_special.png')

            water_mask = (pred == 21)
            sea_mask = (pred == 26)
            river_mask = (pred == 60)
            pool_mask = (pred == 109)
            fall_mask = (pred == 113)
            lake_mask = (pred == 128)

            water_mask = (water_mask | sea_mask | river_mask | pool_mask | fall_mask | lake_mask).astype(int)
            sky_mask = (pred == 2).astype(int)
            grass_mask = (pred == 9).astype(int)
            person_mask = (pred == 12).astype(int)

            if smooth_mask:
                water_mask = water_mask.astype(float) * pred_prob
                sky_mask = sky_mask.astype(float) * pred_prob
                grass_mask = grass_mask.astype(float) * pred_prob
                person_mask = person_mask.astype(float) * pred_prob

            water_masks.append(water_mask)
            sky_masks.append(sky_mask)
            grass_masks.append(grass_mask)
            person_masks.append(person_mask)

            pbar.update(1)

        return water_masks, sky_masks, grass_masks, person_masks


def get_resize_list(image, segment_sizes):
    """Gets list of resized images.

    :param image: Image to resize.
    :param segment_sizes: List of sizes to use.

    :return: List of resizes images.
    """
    normalize = transforms.Normalize(
        mean=[0.40384353, 0.45469216, 0.48145765],
        std=[0.00392157])

    img = rgb_bgr(image.clone())
    img = img.type(torch.float32)
    img = normalize(img)

    resized_list = list()

    for size in segment_sizes:
        trans = transforms.Resize(size)
        resized_list.append(trans(img).unsqueeze(dim=0))

    return resized_list
