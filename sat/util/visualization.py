import matplotlib.pyplot as plt
import numpy as np
import torch
import sat.util.conversion as conversion
#from sat.attacks.attack import AdversarialExample
from PIL import Image
from textwrap import wrap


def single_image(img, data_format='nhwc', save=False, file_name=None):
    """Visualize a single image.

       :param img: Image object. Can be a Tensor or PIL Image.
       :param data_format: Format of data. Must be either 'nhwc' or 'nchw'.
       :param save: If true, saves image to disk.
       :param file_name: Name of file when saving to disk.
   """

    if type(img) == Image:
        plt.imshow(img)
        plt.show()
        return

    if type(img) == torch.Tensor:
        x = np.copy(img.detach().cpu())
    else:
        x = img

    if conversion.is_batched(x):
        x = x[0]

    if data_format.lower() == 'nchw':
        x = conversion.nchw_to_nhwc(x)

    height, width = x.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(1, figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.set_dpi(100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(x)
    ax.axis('off')

    if save:
        assert file_name is not None
        plt.savefig(file_name, pad_inches=0, )

    plt.show()


def adversarial_delta(adv,
                      show_comparison: bool = False,
                      show_labels: bool = False,
                      idx2label=None):
    """Visualize the difference between an original image and adversarial.

    :param adv: Adversarial example object.
    :param show_comparison: If true, visualizes both the original and adversarial image for comparison.
    :param show_labels: If true, shows labels of original and adversarial image.
    :param idx2label: List or Dict connecting label indices to label strings.
    """

    # Copy images
    original = adv.image_ori.detach().cpu().clone()
    adversarial = adv.image_adv.detach().cpu().clone()

    # Calculate perturbation
    delta = np.abs(np.subtract(conversion.nchw_to_nhwc(adversarial), conversion.nchw_to_nhwc(original)))

    if show_comparison:
        fig = plt.figure()
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0, top=1)

        # Visualize original image
        fig.add_subplot(1, 3, 1)
        plt.imshow(conversion.nchw_to_nhwc(original))
        plt.axis('off')
        if idx2label is not None:
            title = "Ori: " + "\n".join(wrap(idx2label[adv.label_ori], 18))
        else:
            title = str(adv.label_ori)
        plt.title(title) if show_labels else None

        # Visualize perturbation
        fig.add_subplot(1, 3, 2)
        plt.imshow(delta)
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.25
        plt.axis('off')
        plt.title('Perturbation')

        # Visualize adversarial image
        fig.add_subplot(1, 3, 3)
        plt.imshow(conversion.nchw_to_nhwc(adversarial))
        plt.axis('off')
        if idx2label is not None:
            title = "Adv: " + "\n".join(wrap(idx2label[adv.label_adv], 18))
        else:
            title = str(adv.label_adv)
        plt.title(title) if show_labels else None
    else:
        plt.imshow(delta)
        plt.axis('off')

    plt.show()



