from sat.util.visualization import adversarial_delta
from sat.util.conversion import unnormalize
import torch
from torchvision import transforms


class Attack:
    """
        Basic attack class used by all implemented adversarial attacks.
    """

    def __init__(self, classifier, device='cpu', norm=None):
        """

        :param classifier: The model to trick
        :param device: 'cuda' if running on gpu, 'cpu' otherwise
        :param norm: Data normalization transformation function.
        """
        self.model = classifier
        self.device = device
        self.list = list()
        self.scorelist = dict()

        if norm is None:
            self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        else:
            self.norm = norm

    def parse_adversarials(self,
                           x: torch.Tensor,
                           y: torch.Tensor,
                           adv_status: torch.Tensor,
                           images=None,
                           include_unsuccessful: bool = False,
                           stop_on_success: bool = True,
                           generate_samples: bool = False
                           ):
        """Parses perturbed images find which of them are adversarial

        :param x: Tensor of perturbed images.
        :param y: Tensor of original labels for images.
        :param adv_status: Tensor defining which adversarial images have already been added. Used to avoid multiple
            adversarials for the same original image.
        :param images: Tensor of original images. Only used to enable visualization of adversarial images.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.
        :param stop_on_success: If true, only one adversarial image will be generated per original image.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.

        :return: True if adversarials have been generated for all images in batch, False otherwise.
        """

        # Check classification of perturbed images
        y_pred = self.model(self.norm(x))
        _, predicted = torch.max(y_pred, 1)

        correct_pred = torch.eq(y.to(self.device), predicted.to(self.device))

        # All missclassified images are added to adversarial list
        for j in range(len(correct_pred)):
            correctly_classified = correct_pred[j].item()

            # Determine if adversarial should be included in final list
            should_include = adv_status[j].item() or not stop_on_success
            should_include = not correctly_classified and should_include
            should_include = include_unsuccessful or should_include

            if not should_include:
                continue

            if generate_samples:
                sample = AdversarialSample(x[j], y[j].item())
            else:
                sample = AdversarialExample(images[j], x[j], y[j].item(), predicted[j].item())

            self.list.append(sample)

        # Stop generating adversarials if successful ones have already been produced
        if include_unsuccessful:
            adv_status.bitwise_and_(torch.zeros_like(adv_status, dtype=torch.bool).to(self.device))
        else:
            adv_status.bitwise_and_(correct_pred)

        if torch.all(adv_status.eq(False)).item() and stop_on_success:
            return True
        else:
            return False

    def parse_adversarials_by_score(self,
                           x: torch.Tensor,
                           y: torch.Tensor,
                           scores,
                           top_scores,
                           images=None,
                           include_unsuccessful: bool = False,
                           generate_samples: bool = False
                           ):
        """Parses perturbed images find which of them are adversarial. Uses scores to find minimal perturbations.

        :param x: Tensor of perturbed images.
        :param y: Tensor of original labels for images.
        :param scores: List of perturbation scores for input images. Smaller perturbations yields smaller scores.
        :param top_scores: Dict of current best adversarials along with their score.
        :param images: Tensor of original images. Only used to enable visualization of adversarial images.
        :param include_unsuccessful: If true, perturbed images that are not adversarial
            will still be included in final list of generated adversarials.
        :param generate_samples: If true, adversarial objects will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.

        """

        # Check classification of perturbed images
        y_pred = self.model(self.norm(x))
        _, predicted = torch.max(y_pred, 1)

        correct_pred = torch.eq(y.to(self.device), predicted)

        # All missclassified images are added to adversarial list
        for j in range(len(correct_pred)):
            correctly_classified = correct_pred[j].item()
            score = scores[j]
            top_score = top_scores.get(j, {'score': float('inf')}).get('score')

            should_include = include_unsuccessful or not correctly_classified
            should_include = should_include and score < top_score

            if not should_include:
                continue

            top_scores[j] = scores[j]

            if generate_samples:
                sample = AdversarialSample(x[j], y[j].item())
            else:
                sample = AdversarialExample(images[j], x[j], y[j].item(), predicted[j].item())

            top_scores.update({j: {'sample': sample, 'score': score}})

    def verify_and_add_adversarial(self, adv, ori, label, generate_samples=False):
        """Simplified version of 'parse_adversarials' function where only one adversarial is considered at a time.

        :param adv: Tensor of perturbed image
        :param ori: Tensor of original image.
        :param label: The label of the original image.
        :param generate_samples: If true, adversarial object will not include original image and labels for
            visualization purposes. Mainly used for adversarial training to save space.

        :return: True if perturbed image is adversarial, False otherwise.
        """

        y_pred = self.model(self.norm(adv))
        _, predicted = torch.max(y_pred, 1)

        if predicted[0].item() == label:
            return False

        if generate_samples:
            sample = AdversarialSample(adv, label)
        else:
            sample = AdversarialExample(ori, adv, label, predicted[0].item())

        self.list.append(sample)
        return True

    def add_best_adversarials(self, adv_dict):
        """Takes dict of adversarials from 'parse_adversarials_by_score' and adds them to the adversarial list."""
        for value in adv_dict.values():
            self.list.append(value.get('sample'))

    def _set_adv_list(self, adv_list):
        """Updates adversarial list. Useful for reloading checkpoints of adversarial dataset generators."""
        self.list = adv_list


class AdversarialSample:
    """Represents an adversarial sample for adversarial training."""

    def __init__(self, image_adv, label_ori):
        """

        :param image_adv: Tensor of adversarial image.
        :param label_ori: True class for the adversarial image.
        """

        if image_adv.ndim == 4:
            image_adv = image_adv.squeeze()

        self.image_adv = image_adv.cpu()
        self.label_ori = label_ori


class AdversarialExample(AdversarialSample):
    """Represents an adversarial example for visualization purposes."""

    def __init__(self, image_ori, image_adv, label_ori, label_adv):
        """

        :param image_adv: Tensor of adversarial image.
        :param image_ori: Tensor of original image.
        :param label_ori: Original image class.
        :param label_adv: Adversarial image class.
        """
        super(AdversarialExample, self).__init__(image_adv, label_ori)

        if image_ori.ndim == 4:
            image_ori = image_ori.squeeze()

        self.image_ori = image_ori.cpu()
        self.label_adv = label_adv

    def unnormalize(self, mean, std, inline=True):
        """Reverts image normalization for mean and std."""
        if inline:
            self.image_ori = unnormalize(self.image_ori, mean, std)
            self.image_adv = unnormalize(self.image_adv, mean, std)
        else:
            return unnormalize(self.image_ori, mean, std), \
                   unnormalize(self.image_adv, mean, std)

    def visualize(self, idx2label=None):
        """Visualizes the adversarial example using matplotlib.

        :param idx2label: List or Dict connecting label indices to label strings.
        """
        adversarial_delta(self, show_comparison=True, show_labels=True, idx2label=idx2label)

    def l0(self):
        """Computes the l0 metric for the adversarial example."""
        return torch.cdist(self.image_ori, self.image_adv, 0)

    def l2(self):
        """Computes the l2 metric for the adversarial example."""
        return torch.cdist(self.image_ori, self.image_adv, 2)

    def linf(self):
        """Computes the l-infinity metric for the adversarial example."""
        return torch.cdist(self.image_ori, self.image_adv, float("inf"))
