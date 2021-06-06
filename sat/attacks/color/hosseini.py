from sat.attacks.attack import Attack
from sat.attacks.permutation import permute_color_channel
import sat.util.conversion as conversion
import matplotlib.colors
import numpy as np
import torch


class HSVAttack(Attack):
    """Implementation of the HSV attack

    Original paper: Semantic Adversarial Examples
    by Hossein Hosseini and Radha Poovendran.

    https://github.com/HosseinHosseini/Semantic-Adversarial-Examples
    """

    def __init__(self, classifier, device='cpu', norm=None):
        """

        :param classifier: The model to fool.
        :param device: 'cuda' if running on gpu, 'cpu' otherwise.
        :param norm: Data normalization transformation function.
        """
        super(HSVAttack, self).__init__(classifier, device, norm)

    def __call__(self,
                 images,
                 labels,
                 max_trials: int = 100,
                 by_score: bool = False,
                 stop_on_success: bool = True,
                 generate_samples: bool = False,
                 include_unsuccessful: bool = False):
        """Performs the attack.

        :param images: Images to perturb.
        :param labels: Labels of original images.
        :param max_trials: Maximum number of trials to find an adversarial image.
        :param by_score: If true, will find the best adversarial with minimum perturbation.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.
        :param stop_on_success: If true, only one adversarial image will be generated per original image.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.

        :return: List of adversarial images
        """

        # Convert to numpy format and hsv color space
        x = np.copy(images)
        x = conversion.nchw_to_nhwc(x)
        x = matplotlib.colors.rgb_to_hsv(x)

        #top_scores = torch.full([len(x)], float('inf'))
        top_scores = dict()
        adv_status = torch.ones(len(x), dtype=torch.bool).to(self.device)

        for i in range(max_trials):

            x_adv = np.copy(x)

            # Choose new hue and saturation values from uniform distribution
            h = np.random.uniform(0, 1, size=(x_adv.shape[0], 1))
            s = np.random.uniform(-1, 1, size=(x_adv.shape[0], 1)) * float(i) / max_trials

            # Permute hue and saturation channels with new values
            permute_color_channel(x_adv, h, 0)
            permute_color_channel(x_adv, s, 1, invalid_value_correction='clip')

            # Revert back to rgb color space and prevent incorrect values
            x_adv = matplotlib.colors.hsv_to_rgb(x_adv)
            x_adv = np.clip(x_adv, 0, 1)

            # Convert back to pytorch datatype
            x_adv = conversion.nhwc_to_nchw(x_adv)
            x_adv = torch.Tensor(x_adv).to(self.device)

            scores = [abs(h[j][0]) + abs(s[j][0]) for j in range(len(x_adv))]

            if by_score:
                self.parse_adversarials_by_score(x_adv,
                                                 labels,
                                                 scores,
                                                 top_scores,
                                                 images,
                                                 include_unsuccessful,
                                                 generate_samples)

            elif self.parse_adversarials(x_adv,
                                           labels,
                                           adv_status,
                                           images,
                                           include_unsuccessful,
                                           stop_on_success,
                                           generate_samples):
                break

        if by_score:
            self.add_best_adversarials(top_scores)

        return self.list



