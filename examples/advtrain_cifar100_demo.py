import torch
import torchvision.transforms as transforms
from sat.attacks.color import HSVAttack
from sat.attacks.geotrans import RotationTranslationAttack
from sat.attacks.manipulate import EdgeFool
from sat.defence import AdversarialTraining
from sat.util import load_model, load_adv_list
from examples.utils.resnet_cifar import resnet50
from examples.utils.data import dataset_loader
from examples.utils.cifar100_trainer import train_cifar100

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    transform_test = transforms.Compose([
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    transform_none = transforms.Compose([
        transforms.ToTensor(),
    ])

    cifar_norm = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    # Load baseline cifar100 classifier
    model_baseline = resnet50().to(device)
    model_baseline = load_model(model_baseline, 'resnet50_cifar100_baseline.pth')
    model_baseline.eval()

    # Load cifar100 dataset
    trainloader, testloader = dataset_loader('cifar100',
                                             batch_size=128,
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
            'max_iterations': 1000,
            'generate_samples': True
        }
    }

    # Set adversarial training parameters
    advtrain = AdversarialTraining(model_baseline,
                                   [HSVAttack, RotationTranslationAttack],
                                   args,
                                   device=device,
                                   norm=cifar_norm,
                                   transforms_train=transform_train,
                                   transforms_test=transform_test)

    edgefool_set = load_adv_list('advset_cifar100_edgefool_test.pth')
    adv_list = advtrain.generate(trainloader,
                                 early_stopping=True,
                                 num_adv=25000,
                                 to_file=False,
                                 file_name='advlist_cifar',
                                 resume=False)

    adv_trainloader, adv_valloader = advtrain.make_loader([adv_list, edgefool_set],
                                                          trainloader,
                                                          val_ratio=0.1,
                                                          fixed_dataset_size=False,
                                                          to_file=False,
                                                          dataset_name='dataset_name')

    # Train model
    train_cifar100(adv_trainloader, adv_valloader, 'resnet50_cifar100_ensemble', resume=False)
