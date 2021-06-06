from sat.attacks.attack import Attack
from torchvision.transforms import functional as F
import torch
import itertools
import random


class RotationTranslationAttack(Attack):
    """Implementation of the RT attack

    Original paper: Exploring the Landscape of Spatial Robustness
    by Logan Engstrom, Brandon Tran and Dimitris Tsipras.

    """

    def __init__(self, classifier, device='cpu', norm=None):
        """

        :param classifier: The model to fool.
        :param device: 'cuda' if running on gpu, 'cpu' otherwise.
        :param norm: Data normalization transformation function.
        """
        super(RotationTranslationAttack, self).__init__(classifier, device, norm)

    def __call__(self,
                 images,
                 labels,
                 mode: str = 'grid_search',
                 k: int = None,
                 max_rot: int = 180,
                 max_trans: int = 50,
                 generate_samples=False,
                 stop_on_success=True,
                 include_unsuccessful=False,
                 by_score: bool = False):

        """Performs the attack

        :param images: Images to perturb.
        :param labels: Labels of original images.
        :param mode: Which approach to use to find adversarials.
            Acceptable inputs are 'grid_search' and 'worst_of_k'.
        :param k: The value of k when using 'worst_of_k'
        :param max_rot: The maximum allowed rotation in degrees.
        :param max_trans: The maximum allowed translation in percentage of height and width.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.
        :param stop_on_success: If true, only one adversarial image will be generated per original image.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.
        :param by_score: If true, will find the best adversarial with minimum perturbation.

        :return: List of adversarial images.
        """

        assert images.ndim == 4
        assert max_rot <= 180
        assert max_trans <= 90

        height, width = images.shape[2:]
        max_dx = max_trans * width // 100
        max_dy = max_trans * height // 100

        if mode == 'worst_of_k':
            assert k is not None

        images = images.to(self.device)
        labels = labels.to(self.device)

        if mode == 'grid_search':
            self.grid_search(images,
                             labels,
                             max_rot,
                             max_dx,
                             max_dy,
                             by_score=by_score,
                             generate_samples=generate_samples,
                             stop_on_success=stop_on_success,
                             include_unsuccessful=include_unsuccessful)
        elif mode == 'worst_of_k':
            self.worst_of_k(images,
                            labels,
                            k,
                            max_rot,
                            max_dx,
                            max_dy,
                            generate_samples=False,
                            stop_on_success=True,
                            include_unsuccessful=include_unsuccessful,
                            by_score=by_score)

        return self.list

    def grid_search(self,
                    images,
                    labels,
                    max_rot,
                    max_dx,
                    max_dy,
                    by_score=False,
                    generate_samples=False,
                    stop_on_success=True,
                    include_unsuccessful=False):
        """Finds adversarial examples by searching all possible perturbations.

        :param images: Images to perturb.
        :param labels: Labels of original images.
        :param max_rot: The maximum allowed rotation in degrees.
        :param max_dx: The maximum allowed translation of width in pixels.
        :param max_dy: The maximum allowed translation of height in pixels.
        :param by_score: If true, will find the best adversarial with minimum perturbation.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.
        :param stop_on_success: If true, only one adversarial image will be generated per original image.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.

        """

        rotation_range = range(-max_rot, max_rot)
        dx_range = range(-max_dx, max_dx)
        dy_range = range(-max_dy, max_dy)
        cartesian_product = [i for i in itertools.product(rotation_range, dx_range, dy_range)]

        def get_abs_sum(arg):
            return abs(arg[0]) + abs(arg[1]) + abs(arg[2])

        if by_score:
            cartesian_product.sort(key=get_abs_sum)
        else:
            random.shuffle(cartesian_product)

        adv_status = torch.ones(images.shape[0], dtype=torch.bool).to(self.device)

        for args in cartesian_product:
            print(args)
            advs = F.affine(images, args[0], [args[1], args[2]], 1.0, [0.0, 0.0])
            if self.parse_adversarials(advs,
                                       labels,
                                       adv_status,
                                       images,
                                       include_unsuccessful=include_unsuccessful,
                                       generate_samples=generate_samples,
                                       stop_on_success=stop_on_success):
                break

    def worst_of_k(self,
                   images,
                   labels,
                   k,
                   max_rot,
                   max_dx,
                   max_dy,
                   generate_samples=False,
                   stop_on_success=True,
                   include_unsuccessful=False,
                   by_score=False):
        """Finds adversarial examples using random values k.

        :param images: Images to perturb.
        :param labels: Labels of original images.
        :param k: The value of k when using 'worst_of_k'.
        :param max_rot: The maximum allowed rotation in degrees.
        :param max_dx: The maximum allowed translation of width in pixels.
        :param max_dy: The maximum allowed translation of height in pixels.
        :param by_score: If true, will find the best adversarial with minimum perturbation.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.
        :param stop_on_success: If true, only one adversarial image will be generated per original image.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.

        """

        adv_status = torch.ones(images.shape[0], dtype=torch.bool).to(self.device)
        top_scores = dict()

        for _ in range(k):
            rot = random.randint(-max_rot, max_rot)
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            advs = F.affine(images, rot, [dx, dy], 1.0, [0.0, 0.0])

            scores = [abs(rot) + abs(dx) + abs(dy) for _ in range(len(images))]

            if by_score:
                self.parse_adversarials_by_score(advs,
                                                 labels,
                                                 scores,
                                                 top_scores,
                                                 images,
                                                 generate_samples=generate_samples,
                                                 include_unsuccessful=include_unsuccessful)

            elif self.parse_adversarials(advs,
                                         labels,
                                         adv_status,
                                         images,
                                         generate_samples=generate_samples,
                                         stop_on_success=stop_on_success,
                                         include_unsuccessful=include_unsuccessful):
                break

        if by_score:
            self.add_best_adversarials(top_scores)
