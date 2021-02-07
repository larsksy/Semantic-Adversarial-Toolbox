from sat.attacks.attack import ColorAttack, Adversarial, AdversarialSet
from sat.attacks.permutation import permute_color_channel
import sat.util.visualization as visualization
import sat.util.conversion as conversion
import PIL as pil
import matplotlib.colors
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sat.util.classes import get_class


class HSVAttack(ColorAttack):

    def __init__(self, classifier, image, label):
        self.classifier = classifier
        self.x = image
        self.y = label
        self.adv_set = AdversarialSet()


    def generate(self, max_trials=10):

        x = np.copy(self.x)
        x = conversion.nchw_to_nhwc(x)
        x = matplotlib.colors.rgb_to_hsv(x)

        for i in range(max_trials):

            x_adv = np.copy(x)
            h = np.random.uniform(0, 1, size=(x_adv.shape[0], 1))
            s = np.random.uniform(-1, 1, size=(x_adv.shape[0], 1)) * float(i) / max_trials

            permute_color_channel(x_adv, h, 0)
            permute_color_channel(x_adv, s, 1, invalid_value_correction='clip')

            x_adv = matplotlib.colors.hsv_to_rgb(x_adv)
            x_adv = np.clip(x_adv, 0, 1)

            x_adv = conversion.nhwc_to_nchw(x_adv)
            x_adv = torch.Tensor(x_adv)

            y_pred = self.classifier(x_adv)
            y_pred = torch.argmax(y_pred)

            if y_pred.item() != self.y.item():
                self.adv_set.add_adversarial(Adversarial(self.x, x_adv, self.y.item(), y_pred.item()))


    def visualize(self):

        for adv in self.adv_set.set:
            #visualization.single_image(img, data_format='nchw')
            visualization.adversarial_delta(adv, show_comparison=True, show_labels=True)



