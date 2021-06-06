'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from sat.util import get_tqdm, load_checkpoint, save_checkpoint, save_model
import sat
import torchvision.transforms as transforms
from sat.attacks.color import HSVAttack
from sat.attacks.geotrans import RotationTranslationAttack
from sat.attacks.manipulate import EdgeFool
from sat.defence import AdversarialTraining
from sat.util import load_model, load_adv_list, load_adv_dataset
from examples.utils.resnet_cifar import resnet50
from examples.utils.data import dataset_loader
from matplotlib import pyplot as plt
import os


def train_cifar10(trainloader, testloader, model_name, resume=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet50(num_classes=10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    best_acc = 0.0
    epoc = 1
    epochs = 200
    adv_list_edgefool = load_adv_list('advlist_cifar10_edgefool_test_ro.pth')
    adv_list_hsv = load_adv_list('advlist_cifar10_hsv_test_ro.pth')
    adv_list_rt = load_adv_list('advlist_cifar10_rt_test_ro.pth')
    adv_list = adv_list_rt + adv_list_hsv + adv_list_edgefool
    advtraining = AdversarialTraining(model_baseline,
                                      [HSVAttack, RotationTranslationAttack],
                                      args,
                                      device=device,
                                      norm=cifar_norm,
                                      transforms_train=transform_train_adv,
                                      transforms_test=transform_test_adv)

    training_loss = list()
    standard_loss = list()
    adversarial_loss = list()

    total = epochs * (len(trainloader.dataset) + len(testloader.dataset) + len(adv_list))
    t = get_tqdm('Training CIFAR10 Model', total, purge=True)

    if resume:
        model, train_scheduler, optimizer, epoc, best_acc = load_checkpoint(
            model, scheduler, optimizer, model_name + '_ckpt.pth')
        t.update((epoc - 1) * (len(trainloader.dataset) + len(testloader.dataset) + len(adv_list)))
        training_loss = torch.load(os.path.join(os.getcwd(), 'train_loss2.pth'))
        standard_loss = torch.load(os.path.join(os.getcwd(), 'standard_loss2.pth'))
        adversarial_loss = torch.load(os.path.join(os.getcwd(), 'adv_loss2.pth'))

    for epoch in range(epoc, epochs + 1):
        train_acc = sat.train(model, trainloader, loss_function, optimizer, device=device, t=t, best_acc=best_acc)
        acc = sat.test(model, testloader, loss_function, device=device, t=t, best_acc=best_acc)
        adv_acc = advtraining.test(model, adv_list, should_generate=False, t=t)
        training_loss.append(train_acc)
        standard_loss.append(acc)
        adversarial_loss.append(adv_acc)
        scheduler.step()

        if best_acc < acc:
            save_checkpoint(model, scheduler, optimizer, epoch + 1, acc, model_name + '_ckpt.pth')
            torch.save(training_loss, os.path.join(os.getcwd(), 'train_loss2.pth'))
            torch.save(standard_loss, os.path.join(os.getcwd(), 'standard_loss2.pth'))
            torch.save(adversarial_loss, os.path.join(os.getcwd(), 'adv_loss2.pth'))
            best_acc = acc

    save_model(model, model_name + '.pth')
    torch.save(training_loss, os.path.join(os.getcwd(), 'train_loss2.pth'))
    torch.save(standard_loss, os.path.join(os.getcwd(), 'standard_loss2.pth'))
    torch.save(adversarial_loss, os.path.join(os.getcwd(), 'adv_loss2.pth'))


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_train_adv = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test_adv = transforms.Compose([
        #transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_none = transforms.Compose([
        transforms.ToTensor(),
    ])

    cifar_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Load baseline cifar100 classifier
    model_baseline = resnet50(num_classes=10).to(device)
    model_baseline = load_model(model_baseline, 'resnet50_cifar10_baseline.pth')
    model_baseline.eval()

    # Load cifar100 dataset
    trainloader, testloader = dataset_loader('cifar10',
                                             batch_size=128,
                                             transforms_train=transform_none,
                                             transforms_test=transform_none,
                                             shuffle=False,
                                             num_workers=0)

    trainloader_clean, testloader_clean = dataset_loader('cifar10',
                                                         batch_size=128,
                                                         transforms_train=transform_train,
                                                         transforms_test=transform_test,
                                                         shuffle=False,
                                                         num_workers=0)


    args = {
        'RotationTranslationAttack': {
            'k': 100,
            'mode': 'worst_of_k',
            'max_rot': 30,
            'max_trans': 20,
            'generate_samples': True,
            'stop_on_success': True,
            'include_unsuccessful': False,
            'by_score': True
        },
        'HSVAttack': {
            'generate_samples': True,
            'stop_on_success': True,
            'include_unsuccessful': False,
            'max_trials': 100,
            'by_score': True
        },
        'EdgeFool': {
            'generate_samples': True,
            'max_iterations': 1000,
            'break_threshold': 0.0005,
            'accept_threshold': 0.005,
            'include_unsuccessful': False,
            'verbose': False
        },
        'ColorFool': {
            'max_iterations': 700,
            'generate_samples': True
        }
    }

    torch.manual_seed(0)

    # Set adversarial training parameters
    advtrain = AdversarialTraining(model_baseline,
                                   [HSVAttack, RotationTranslationAttack],
                                   args,
                                   device=device,
                                   norm=cifar_norm,
                                   transforms_train=transform_train_adv,
                                   transforms_test=transform_test_adv)

    #adv_list = advtrain.generate(trainloader,
    #                             early_stopping=True,
    #                             num_adv=25000,
    #                             to_file=False,
    #                             file_name='advlist_cifar',
    #                             resume=False)

    #adv_list_edgefool = load_adv_list('advlist_cifar10_edgefool_train.pth')
    #adv_trainloader, adv_valloader = advtrain.make_loader([adv_list, adv_list_edgefool],
    #                                                      trainloader,
    #                                                      val_ratio=0.001,
    #                                                      fixed_dataset_size=False,
    #                                                      to_file=True,
    #                                                      dataset_name='advset_cifar10_graph')

    # adv_trainloader = load_adv_dataset('advset_cifar10_ensemble_train.pth')
    # adv_valloader = load_adv_dataset('advset_cifar10_ensemble_val.pth')
    #adv_trainloader = load_adv_dataset('advset_cifar10_graph_train.pth')

    # Train model
    #train_cifar10(adv_trainloader, testloader_clean, 'model_name', resume=True)

    training_loss = torch.load(os.path.join(os.getcwd(), 'train_loss2.pth'))
    standard_loss = torch.load(os.path.join(os.getcwd(), 'standard_loss2.pth'))
    adversarial_loss = torch.load(os.path.join(os.getcwd(), 'adv_loss2.pth'))

    print(len(training_loss))
    print(len(standard_loss))
    print(len(adversarial_loss))

    plt.plot(training_loss, label='Training accuracy')
    plt.plot(standard_loss, label='Standard accuracy')
    plt.plot(adversarial_loss, label='Adversarial accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


