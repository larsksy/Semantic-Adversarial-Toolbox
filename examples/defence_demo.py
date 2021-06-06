from PIL import Image
import torchvision.models as models
import torch
import torchvision.transforms as transforms
from sat.defence import FeatureSqueezing, JPEGCompression
from sat.util import single_image, load_model, load_adv_list_checkpoint
from examples.utils.data import image_loader, dataset_loader
import sat
import numpy as np


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_none = transforms.Compose([
        transforms.ToTensor(),
    ])

    imagenet_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    trainloader, testloader = dataset_loader('cifar10',
                                             batch_size=128,
                                             transforms_train=transform_none,
                                             transforms_test=transform_none,
                                             shuffle=False,
                                             num_workers=0)


    # Load model
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # Load example image
    image = Image.open("./data/custom/dog.jpg")
    image = image_loader(image, transforms=transform_none)

    # Fetch classification for clean image
    y_pred = model(imagenet_norm(image.to(device)))
    _, y_pred = torch.max(y_pred, 1)

    # Visualize original image
    single_image(image, data_format='nchw', save=False)

    # Feature Squeezing Defence
    featureSqueezing = FeatureSqueezing(bit_depth=3, kernel_size=3)
    single_image(featureSqueezing(image), data_format='nchw', save=False)

    # JPEG Compression Defence
    jpegcomp = JPEGCompression(quality=20)
    single_image(jpegcomp(image), data_format='nchw', save=False)

    # Apply defence to all images in dataloader
    testloader = featureSqueezing.apply(testloader)

