'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim

from examples.utils.resnet_cifar import resnet50
from sat.util import get_tqdm, load_checkpoint, save_checkpoint, save_model
import sat


def train_cifar10(trainloader, testloader, model_name, resume=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet50(num_classes=10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    best_acc = 0.0
    epoc = 1
    epochs = 200

    total = epochs * (len(trainloader.dataset) + len(testloader.dataset))
    t = get_tqdm('Training CIFAR10 Model', total, purge=True)

    if resume:
        model, train_scheduler, optimizer, epoc, best_acc = load_checkpoint(
            model, scheduler, optimizer, model_name + '_ckpt.pth')
        t.update((epoc - 1) * (len(trainloader.dataset) + len(testloader.dataset)))

    for epoch in range(epoc, epochs + 1):
        sat.train(model, trainloader, loss_function, optimizer, device=device, t=t, best_acc=best_acc)
        acc = sat.test(model, testloader, loss_function, device=device, t=t, best_acc=best_acc)
        scheduler.step()

        if best_acc < acc:
            save_checkpoint(model, scheduler, optimizer, epoch + 1, acc, model_name + '_ckpt.pth')
            best_acc = acc

    save_model(model, model_name + '.pth')
