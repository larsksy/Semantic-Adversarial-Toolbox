import numpy as np
import torch
from torch import nn, optim, autograd
from sat.util import single_image, nhwc_to_nchw, nchw_to_nhwc
from sat.models import DeepGuidedFilter
from sat.attacks.manipulate.util import detail_enhance_lab
from sat.attacks.attack import Attack, AdversarialExample, AdversarialSample
from sat.util.store import save_model, load_model
from scipy.fftpack import fft2, ifft2


class EdgeFool(Attack):
    """Implementation of the EdgeFool attack

    Original paper: EdgeFool: An Adversarial Image Enhancement Filter
    by Ali Shahin Shamsabadi, Changjae Oh and Andrea Cavallaro.

    https://github.com/smartcameras/EdgeFool
    """


    def __init__(self, classifier, device='cpu', norm=None):
        """

        :param classifier: The model to fool.
        :param device: 'cuda' if running on gpu, 'cpu' otherwise.
        :param norm: Data normalization transformation function.
        """
        super(EdgeFool, self).__init__(classifier, device, norm)
        self.should_load_filter = False

    def __call__(self,
                 images,
                 labels,
                 generate_samples=False,
                 include_unsuccessful=False,
                 max_iterations=500,
                 break_threshold=0.0005,
                 accept_threshold=0.01,
                 verbose=False):
        """Performs the attack

        :param images: Images to perturb.
        :param labels: Labels of original images.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.
        :param max_iterations: Maximum number of trials to find an adversarial image.
        :param break_threshold: Loss threshold for moving on to next image
        :param accept_threshold: Loss threshold for accepting an adversarial image.
        :param verbose: If true, will print loss at every iteration.

        :return:
        """
        self.model.eval()

        guided_filter = DeepGuidedFilter().to(self.device)

        if self.should_load_filter:
            guided_filter = load_model(guided_filter, 'guided_filter.pth')

        # Smoothing loss function
        criterion = nn.MSELoss().to(self.device)

        # Setup optimizer
        optimizer = optim.Adam(guided_filter.parameters(), lr=0.001)

        # Freeze the parameters of the classifeir under attack to not be updated
        for param in self.model.parameters():
            param.requires_grad = False

        # Pre-processing the original image
        images = images.to(self.device)

        for i in range(len(images)):
            smooth_image = _l0_minimization(images[i],
                                      lmd=0.02,
                                      beta_max=1e5,
                                      beta_rate=2.0,
                                      max_iter=30)

            smooth_image = np.clip(smooth_image, 0, 1)

            # Visualize smooth image
            #single_image(smooth_image, save=True, file_name='edgefool_special.png')

            adv = self.train(images[i],
                             smooth_image,
                             labels[i].item(),
                             guided_filter,
                             criterion,
                             optimizer,
                             generate_samples=generate_samples,
                             max_iterations=max_iterations,
                             break_threshold=break_threshold,
                             accept_threshold=accept_threshold,
                             include_unsuccessful=include_unsuccessful,
                             verbose=verbose)

            if adv is not None:
                self.list.append(adv)
                self.should_load_filter = True
            else:
                self.should_load_filter = False

        return self.list

    def train(self,
              image,
              smooth_img,
              label,
              filter,
              criterion,
              optimizer,
              max_iterations=500,
              generate_samples=False,
              include_unsuccessful=False,
              break_threshold=0.0005,
              accept_threshold=0.01,
              verbose=False):
        """Finds a single adversarial image using EdgeFool.

        :param image: Image to perturb.
        :param smooth_img: Smoothened version of input image to use for edge comparison.
        :param label: Label of original image.
        :param filter: The image filter model.
        :param criterion: Loss function to calculate edge loss.
        :param optimizer: The optimizer for training the filter.
        :param max_iterations: Maximum number of trials to find an adversarial image.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.
        :param break_threshold: Loss threshold for moving on to next image
        :param accept_threshold: Loss threshold for accepting an adversarial image.
        :param verbose: If true, will print loss at every iteration.

        :return: Adversarial image
        """
        adv = None
        best_loss = float('inf')

        smooth_img = nhwc_to_nchw(torch.Tensor(smooth_img).unsqueeze(dim=0)).to(self.device)
        image = image.unsqueeze(dim=0)

        for it in range(max_iterations):

            # Smooth images
            x_smooth = filter(image, smooth_img)

            # Enhance adversarial image
            enh = detail_enhance_lab(image, x_smooth)

            # Prediction of the adversarial image using the classifier chosen for attacking
            logit_enh = self.model(self.norm(enh.permute(2, 0, 1).unsqueeze(dim=0)))
            class_enh = torch.argmax(logit_enh).item()

            # Computing smoothing and adversarial losses
            loss1 = criterion(x_smooth, smooth_img)
            loss2 = AdvLoss(logit_enh, label, is_targeted=False, num_classes=logit_enh.shape[1])

            # Combining the smoothing and adversarial losses
            loss = 10 * loss1 + loss2

            if verbose:
                print('iteration %s: %s' % (it, loss1))

            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter.parameters(), 0.01)
            optimizer.step()

            # Save the adversarial image when the classifier is fooled and smoothing loss is less than a threshold
            if (label != class_enh or include_unsuccessful) and loss1 < best_loss and loss1 <= accept_threshold:

                if generate_samples:
                    adv = AdversarialSample(nhwc_to_nchw(enh.detach().cpu()), label)
                else:
                    adv = AdversarialExample(image, nhwc_to_nchw(enh.detach().cpu()), label, class_enh)

                best_loss = loss1
                if loss1 < break_threshold:
                    break

        # Save the FCNN
        save_model(filter, 'guided-filter.pth')
        self.filter_trained = True
        return adv



# Util for l0 gradient minimization used to smoothen image edges
# https://github.com/t-suzuki/l0_gradient_minimization_test
def _l0_minimization(image, lmd=0.02, beta_max=1e5, beta_rate=2.0, max_iter=30):
    """Iteratively smoothens an image using l0 minimization

    :param image: The image to smoothen
    :param lmd: Control variable for calculating smoothness.
    :param beta_max: Threshold for acceptable smoothness
    :param beta_rate: Control variable for calculating smoothness.
    :param max_iter: Maximum number of trials to get acceptably smooth image.

    :return: Smoothened version of image.
    """

    S = np.array(nchw_to_nhwc(image.detach().cpu()))

    F_I = fft2(S, axes=(0, 1))
    Ny, Nx = S.shape[:2]
    D = S.shape[2] if S.ndim == 3 else 1
    dx, dy = np.zeros((Ny, Nx)), np.zeros((Ny, Nx))
    dx[Ny // 2, Nx // 2 - 1:Nx // 2 + 1] = [-1, 1]
    dy[Ny // 2 - 1:Ny // 2 + 1, Nx // 2] = [-1, 1]
    F_denom = np.abs(fft2(dx)) ** 2.0 + np.abs(fft2(dy)) ** 2.0

    if D > 1:
        F_denom = np.dstack([F_denom] * D)

    beta = lmd * 2.0
    for i in range(max_iter):

        # with S, solve for hp and vp in Eq. (12)
        hp, vp = _circulant2_dx(S, 1), _circulant2_dy(S, 1)

        if D == 1:
            mask = hp ** 2.0 + vp ** 2.0 < lmd / beta
        else:
            mask = np.sum(hp ** 2.0 + vp ** 2.0, axis=2) < lmd / beta

        hp[mask] = 0.0
        vp[mask] = 0.0

        # with hp and vp, solve for S in Eq. (8)
        hv = _circulant2_dx(hp, -1) + _circulant2_dy(vp, -1)
        S = np.real(ifft2((F_I + (beta * fft2(hv, axes=(0, 1)))) / (1.0 + beta * F_denom), axes=(0, 1)))

        beta *= beta_rate
        if beta > beta_max:
            break

    return S


def _circulant2_dx(xs, h):
    """Calculates a 2d circular shift differential value in the x direction"""
    stack = np.hstack([xs[:, h:], xs[:, :h]] if h > 0 else [xs[:, h:], xs[:, :h]])
    return stack - xs


def _circulant2_dy(xs, h):
    """Calculates a 2d circular shift differential value in the y direction"""
    stack = np.vstack([xs[h:, :], xs[:h, :]] if h > 0 else [xs[h:, :], xs[:h, :]])
    return stack - xs


def AdvLoss(logits, target, is_targeted, num_classes=1000, kappa=0):
    """Adversarial loss function.

    :param logits: Output prediction from model.
    :param target: Target prediction.
    :param is_targeted: Wether the attack is targeted or not.
    :param num_classes: Number of different output classes.
    :param kappa: Control variable.

    :return: The adversarial loss for the input image.
    """

    # inputs to the softmax function are called logits.
    # https://arxiv.org/pdf/1608.04644.pdf
    target_one_hot = torch.eye(num_classes).type(logits.type())[target]

    # workaround here.
    # subtract large value from target class to find other max value
    # https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py
    real = torch.sum(target_one_hot * logits, 1)
    other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if is_targeted:
        return torch.sum(torch.max(other - real, kappa))
    return torch.sum(torch.max(real - other, kappa))
