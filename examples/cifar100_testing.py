import torch
import torchvision.transforms as transforms
from sat.attacks.color import HSVAttack, ColorFool
from sat.attacks.geotrans import RotationTranslationAttack
from sat.attacks.manipulate import EdgeFool
from sat.defence import AdversarialTraining
from sat.util import load_model, load_adv_list, get_tqdm
from examples.utils.resnet_cifar import resnet50
from examples.utils.data import dataset_loader
import sat

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

    transform_clean = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                             (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])

    cifar_norm = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                      (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))

    criterion = torch.nn.CrossEntropyLoss()

    # Initializing different classifiers to compare results
    model_baseline = resnet50().to(device)
    model_baseline = load_model(model_baseline, 'resnet50_cifar100_baseline.pth')
    model_baseline.eval()

    model_test = resnet50().to(device)
    model_test = load_model(model_test, 'resnet50_cifar100_test.pth')
    model_test.eval()

    model_hsv = resnet50().to(device)
    model_hsv = load_model(model_hsv, 'advtrain_cifar100_HSV.pth')
    model_hsv.eval()

    model_rt = resnet50().to(device)
    model_rt = load_model(model_rt, 'advtrain_cifar100_RT.pth')
    model_rt.eval()

    model_edgefool = resnet50().to(device)
    model_edgefool = load_model(model_edgefool, 'advtrain_cifar100_EdgeFool.pth')
    model_edgefool.eval()

    model_ensemble = resnet50().to(device)
    model_ensemble = load_model(model_ensemble, 'advtrain_cifar100_triple.pth')
    model_ensemble.eval()

    # Dataset without transforms for generating adversarial datasets
    trainloader, testloader = dataset_loader('cifar100',
                                             batch_size=128,
                                             transforms_train=transform_none,
                                             transforms_test=transform_none,
                                             shuffle=False,
                                             num_workers=0)

    # Normalized dataset for testing on unperturbed images
    trainloader_clean, testloader_clean = dataset_loader('cifar100',
                                                         batch_size=256,
                                                         transforms_train=transform_clean,
                                                         transforms_test=transform_clean,
                                                         shuffle=False,
                                                         num_workers=0)

    # Arguments for different adversarial attacks
    args = {
        'RotationTranslationAttack': {
            'k': 100,
            'mode': 'worst_of_k',
            'max_rot': 30,
            'max_trans': 20,
            'generate_samples': True,
            'stop_on_success': True,
            'include_unsuccessful': True,
            'by_score': False
        },
        'HSVAttack': {
            'generate_samples': True,
            'stop_on_success': True,
            'include_unsuccessful': True,
            'max_trials': 100,
            'by_score': False
        },
        'EdgeFool': {
            'generate_samples': True,
            'max_iterations': 10,
            'break_threshold': 0.0005,
            'accept_threshold': 0.005,
            'include_unsuccessful': True,
            'verbose': False
        },
    }

    # Testing on clean testset
    data_len = len(testloader_clean.dataset)
    sat.test(model_test, testloader_clean, criterion, device, t=get_tqdm('Clean testset - No AdvTrain', data_len, True))
    sat.test(model_hsv, testloader_clean, criterion, device, t=get_tqdm('Clean testset - HSV AdvTrain', data_len, True))
    sat.test(model_rt, testloader_clean, criterion, device, t=get_tqdm('Clean testset - RT AdvTrain', data_len, True))
    sat.test(model_edgefool, testloader_clean, criterion, device,
             t=get_tqdm('Clean testset - EdgeFool AdvTrain', data_len, True))
    sat.test(model_ensemble, testloader_clean, criterion, device,
             t=get_tqdm('Clean testset - Ensemble AdvTrain', data_len, True))

    # Testing on adversarial images from HSV Attack
    advtrain = AdversarialTraining(model_baseline,
                                   [HSVAttack],
                                   args,
                                   device=device,
                                   norm=cifar_norm,
                                   transforms_train=transform_train,
                                   transforms_test=transform_test)

    # hsv_set = load_adv_list('advset_cifar100_hsv_test_no_ro.pth')
    hsv_set = advtrain.generate(testloader,
                                early_stopping=True,
                                num_adv=10000,
                                to_file=True,
                                file_name='advlist_cifar100_hsv_test_all.pth',
                                resume=False)

    advtrain.test(model_test, hsv_set, should_generate=False, batch_size=256, t_desc='HSV testset - No AdvTrain')
    advtrain.test(model_hsv, hsv_set, should_generate=False, batch_size=256, t_desc='HSV testset - HSV AdvTrain')
    advtrain.test(model_rt, hsv_set, should_generate=False, batch_size=256, t_desc='HSV testset - RT AdvTrain')
    advtrain.test(model_edgefool, hsv_set, should_generate=False, batch_size=256,
                  t_desc='HSV testset - EdgeFool AdvTrain')
    advtrain.test(model_ensemble, hsv_set, should_generate=False, batch_size=256,
                  t_desc='HSV testset - Ensemble AdvTrain')

    # Testing on adversarial images from RT Attack
    advtrain = AdversarialTraining(model_baseline,
                                   [RotationTranslationAttack],
                                   args,
                                   device=device,
                                   norm=cifar_norm,
                                   transforms_train=transform_train,
                                   transforms_test=transform_test)

    # rt_set = load_adv_list('advset_cifar100_rt_test_no_ro.pth')
    rt_set = advtrain.generate(testloader,
                               early_stopping=True,
                               num_adv=10000,
                               to_file=True,
                               file_name='advlist_cifar100_rt_test_all.pth',
                               resume=False)

    advtrain.test(model_test, rt_set, should_generate=False, batch_size=256, t_desc='RT testset - No AdvTrain')
    advtrain.test(model_hsv, rt_set, should_generate=False, batch_size=256, t_desc='RT testset - HSV AdvTrain')
    advtrain.test(model_rt, rt_set, should_generate=False, batch_size=256, t_desc='RT testset - RT AdvTrain')
    advtrain.test(model_edgefool, rt_set, should_generate=False, batch_size=256,
                  t_desc='RT testset - EdgeFool AdvTrain')
    advtrain.test(model_ensemble, rt_set, should_generate=False, batch_size=256,
                  t_desc='RT testset - Ensemble AdvTrain')

    # Testing on adversarial images from EdgeFool Attack
    advtrain = AdversarialTraining(model_baseline,
                                   [EdgeFool],
                                   args,
                                   device=device,
                                   norm=cifar_norm,
                                   transforms_train=transform_train,
                                   transforms_test=transform_test)

    # Edgefool set is pre-made due to time constraints.
    # edgefool_set = load_adv_list('advset_cifar100_edgefool_test_no_ro.pth')
    edgefool_set = advtrain.generate(testloader,
                                     early_stopping=True,
                                     num_adv=10000,
                                     to_file=True,
                                     file_name='advlist_cifar100_edgefool_test_all.pth',
                                     resume=False)

    advtrain.test(model_test, edgefool_set, should_generate=False, batch_size=256,
                  t_desc='EdgeFool testset - No AdvTrain')
    advtrain.test(model_hsv, edgefool_set, should_generate=False, batch_size=256,
                  t_desc='EdgeFool testset - HSV AdvTrain')
    advtrain.test(model_rt, edgefool_set, should_generate=False, batch_size=256,
                  t_desc='EdgeFool testset - RT AdvTrain')
    advtrain.test(model_edgefool, edgefool_set, should_generate=False, batch_size=256,
                  t_desc='EdgeFool testset - EdgeFool AdvTrain')
    advtrain.test(model_ensemble, edgefool_set, should_generate=False, batch_size=256,
                  t_desc='EdgeFool testset - Ensemble AdvTrain')
