# train.py
# !/usr/bin/env	python3

""" train network using pytorch
author baiyu
"""


import torch
import torch.nn as nn
import torch.optim as optim

import sat
from examples.utils.resnet_cifar import resnet50
from examples.utils.WarmupScheduler import WarmUpLR
from sat.util import load_checkpoint, save_checkpoint, save_model, get_tqdm


def train_cifar100(trainloader, testloader, model_name, resume=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet50().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    iter_per_epoch = len(trainloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
    best_acc = 0.0
    epoc = 1
    epochs = 200

    total = epochs * (len(trainloader.dataset) + len(testloader.dataset))
    t = get_tqdm('Training', total)

    if resume:
        model, train_scheduler, optimizer, epoc, best_acc = load_checkpoint(
            model, train_scheduler, optimizer, model_name + '_ckpt.pth')
        t.update((epoc - 1) * (len(trainloader.dataset) + len(testloader.dataset)))

    for epoch in range(epoc, epochs + 1):

        if epoch > 1:
            train_scheduler.step(epoch)

        sat.train(model, trainloader, loss_function, optimizer, device=device, t=t, best_acc=best_acc)
        acc = sat.test(model, testloader, loss_function, device=device, t=t, best_acc=best_acc)

        if epoch <= 1:
            warmup_scheduler.step()

        if best_acc < acc:
            save_checkpoint(model, train_scheduler, optimizer, epoch + 1, acc, model_name + '_ckpt.pth')
            best_acc = acc

    save_model(model, model_name + '.pth')




