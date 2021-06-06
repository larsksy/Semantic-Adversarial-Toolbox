from PIL import Image
import torchvision.models as models
import torch
import torchvision.transforms as transforms
from sat.attacks.color import HSVAttack, ColorFool
from sat.attacks.geotrans import RotationTranslationAttack
from sat.attacks.manipulate import EdgeFool
from sat.util.visualization import single_image
from examples.utils.data import image_loader
from examples.utils.classes import get_class


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_none = transforms.Compose([
        #transforms.Resize(224),
        transforms.ToTensor(),
    ])

    imagenet_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # get idx2label for imagenet
    idx2label = get_class(dataset='imagenet')

    # Load model
    model = models.resnet50(pretrained=True).to(device)
    model.eval()

    # Load example image
    image = Image.open("./data/custom/dog.jpg")
    image = image_loader(image, transforms=transform_none)

    # Fetch classification for clean image
    y_pred = model(imagenet_norm(image.to(device)))
    _, y_pred = torch.max(y_pred, 1)

    # Demo HSV attack
    hsv = HSVAttack(model, device=device, norm=imagenet_norm)
    hsv(image, y_pred, by_score=False, max_trials=10)[0].visualize(idx2label=idx2label)

    # Demo EdgeFool attack
    edgefool = EdgeFool(model, device=device, norm=imagenet_norm)
    edgefool(torch.Tensor(image), y_pred, verbose=True)[0].visualize(idx2label=idx2label)

    # Demo Rotation and Translation attack
    engstrom = RotationTranslationAttack(model, device=device, norm=imagenet_norm)
    new_image = engstrom(image, y_pred, mode='worst_of_k', k=100, max_trans=10, max_rot=20, by_score=True)
    new_image[0].visualize(idx2label=idx2label)

    # Demo ColorFool attack
    colorfool = ColorFool(model, device=device)
    colorfool(image, y_pred)[0].visualize(idx2label=idx2label)

