import torchvision.transforms as transform
from torch.autograd import Variable
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def image_loader(image_pil, transforms=None):
    """load image, returns cuda tensor"""

    if transforms is None:
        transforms = transform.Compose(
            [transform.Resize(256),
             transform.ToTensor(),
             # transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
             ])

    image = transforms(image_pil).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image


def dataset_loader(dataset_string,
                   transforms_train=None,
                   transforms_test=None,
                   batch_size=8,
                   shuffle=True,
                   num_workers=1):
    """load dataset, returns dataloaders for trainset and testset"""
    if transforms_train is None:
        transforms_train = transform.Compose([transform.ToTensor()])

    if transforms_test is None:
        transforms_test = transform.Compose([transform.ToTensor()])

    if dataset_string == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
    elif dataset_string == 'cifar100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transforms_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transforms_test)
    else:
        raise ValueError('%s is not a valid dataset argument' % dataset_string)

    testloader = DataLoader(testset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return trainloader, testloader
