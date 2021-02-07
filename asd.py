from torchvision import models
import torchvision.transforms as transforms
import torch
from PIL import Image


if __name__ == '__main__':
    resnet = models.resnet50(pretrained=True)

    transform = transforms.Compose([  # [1]
        transforms.Resize(256),  # [2
        transforms.CenterCrop(224),  # [3]
        transforms.ToTensor(),  # [4]
        transforms.Normalize(  # [5]
        mean = [0.485, 0.456, 0.406],  # [6]
        std = [0.229, 0.224, 0.225]
        )])

    img = Image.open("./Dataset/dog.jpg")
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    print(batch_t.shape)

    resnet.eval()

    out = resnet(batch_t)
    print(out.shape)

    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(labels[index[0]], percentage[index[0]].item())

    _, indices = torch.sort(out, descending=True)
    print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])



