import matplotlib.pyplot as plt
import numpy as np
import sat.util.conversion as conversion
from sat.util.classes import get_class
from PIL import Image


def single_image(img, data_format='nhwc'):

    if type(img) == Image:
        plt.imshow(img)
        plt.show()
        return

    x = np.copy(img)

    if conversion.is_batched(x):
        x = x[0]

    if data_format.lower() == 'nchw':
        x = conversion.nchw_to_nhwc(x)

    plt.imshow(x)
    plt.show()


def adversarial_delta(adv, show_comparison=False, show_labels=False):
    original = adv.image.detach().clone()[0]
    adversarial = adv.image_adv.detach().clone()[0]
    #original = conversion.unnormalize(original, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #adversarial = conversion.unnormalize(adversarial, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    delta = np.abs(np.subtract(conversion.nchw_to_nhwc(original), conversion.nchw_to_nhwc(adversarial)))

    if show_comparison:
        fig = plt.figure()

        fig.add_subplot(1, 3, 1)
        plt.imshow(conversion.nchw_to_nhwc(original))
        plt.axis('off')
        plt.title(get_class(adv.label, 'imagenet')) if show_labels else None

        fig.add_subplot(1, 3, 2)
        plt.imshow(delta)
        plt.axis('off')

        fig.add_subplot(1, 3, 3)
        plt.imshow(conversion.nchw_to_nhwc(adversarial))
        plt.axis('off')
        plt.title(get_class(adv.label_adv, 'imagenet')) if show_labels else None
    else:
        plt.imshow(delta)
        plt.axis('off')

    plt.show()



