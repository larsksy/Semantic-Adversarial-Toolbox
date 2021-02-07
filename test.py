from PIL import Image
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transform
from torch.autograd import Variable
import torch
from sat.attacks.color.hosseini import HSVAttack
from sat.data.datasets import TestDataset
from sat.util.classes import get_class
from sat.defense.FeatureSqueezing import FeatureSqueezing
from sat.util.visualization import single_image
import numpy as np


def image_loader(image_pil):
    """load image, returns cuda tensor"""
    loader = transform.Compose([transform.Resize(imsize),
                                transform.ToTensor(),
                                #transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    image = loader(image_pil).float()
    image = Variable(image, requires_grad=False)
    #image = image.numpy()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU


if __name__ == '__main__':
    imsize = 256
    image = Image.open("./Dataset/dog.jpg")
    #image = image.convert('HSV')
    model = models.resnet50(pretrained=True)
    model.eval()
    image = image_loader(image)

    print(image.shape)

    defence = FeatureSqueezing(image, bit_depth=8, kernel_size=2)

    print(defence.squeezed.shape)

    single_image(image, data_format='nchw')
    single_image(defence.squeezed, data_format='nchw')

    y_pred = model(image)
    y_pred = torch.argmax(y_pred)

    #hsv = HSVAttack(model, image, y_pred)
    #hsv.generate()
    #hsv.visualize()
