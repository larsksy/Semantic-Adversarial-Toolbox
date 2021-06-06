import torch
import torchvision.transforms as transforms
from sat.attacks.color import HSVAttack
from sat.attacks.geotrans import RotationTranslationAttack
from sat.attacks.manipulate import EdgeFool
from sat.defence import AdversarialTraining
from sat.util import load_model, load_adv_list, load_adv_dataset
from examples.utils.resnet_cifar import resnet50
from examples.utils.data import dataset_loader
from examples.utils.cifar10_trainer import train_cifar10

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        # transforms.ToTensor(),
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
                                             batch_size=16,
                                             transforms_train=transform_none,
                                             transforms_test=transform_none,
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
                                   transforms_train=transform_train,
                                   transforms_test=transform_test)

    adv_list = advtrain.generate(trainloader,
                                 early_stopping=True,
                                 num_adv=25000,
                                 to_file=False,
                                 file_name='advlist_cifar',
                                 resume=False)

    adv_list_edgefool = load_adv_list('advlist_cifar10_edgefool_train.pth')
    adv_trainloader, adv_valloader = advtrain.make_loader([adv_list, adv_list_edgefool],
                                                          trainloader,
                                                          val_ratio=0.1,
                                                          fixed_dataset_size=False,
                                                          to_file=True,
                                                          dataset_name='advset_cifar10_ensemble')

    # adv_trainloader = load_adv_dataset('advset_cifar10_ensemble_train.pth')
    # adv_valloader = load_adv_dataset('advset_cifar10_ensemble_val.pth')

    # Train model
    train_cifar10(adv_trainloader, adv_valloader, 'model_name', resume=True)
