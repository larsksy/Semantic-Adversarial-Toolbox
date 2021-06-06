import torch
import os


def verify_path(file_name, folder):
    """Ensures folders for checkpoints and network parameters exists.

       :param file_name: Name of file.
       :param folder: Name of folder
    """
    folder_path = os.path.join(os.getcwd(), folder)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    return os.path.join(os.getcwd(), folder, file_name)


def save_model(model, file_name):
    """Saves model parameters to disk.

       :param model: Model to save.
       :param file_name: Name of file to save to.
    """
    path = verify_path(file_name, 'pretrained/')
    torch.save(model.state_dict(), path)


def save_checkpoint(model, scheduler, optimizer, epoch, acc, file_name):
    """Saves model checkpoint during training.

       :param model: Model to save.
       :param scheduler: Scheduler used in training.
       :param optimizer: Optimizer used in training.
       :param epoch: Current epoch in training.
       :param acc: Current best accuracy in training.
       :param file_name: Name of file to save to.
    """

    path = verify_path(file_name, 'checkpoints/')
    dct = {
        'net': model.state_dict(),
        'epoch': epoch,
        'opt': optimizer.state_dict(),
        'sch': scheduler.state_dict(),
        'acc': acc
    }
    torch.save(dct, path)


def save_adv_list_checkpoint(file_name, adv_list, index):
    """Saves checkpoint of generated adversarial images for adversarial training.

       :param file_name: Name of file to save to.
       :param adv_list: List of adversarial images.
       :param index: Current dataset index.
    """

    path = verify_path(file_name, 'checkpoints/')

    dct = {
        'list': adv_list,
        'index': index,
    }
    torch.save(dct, path)


def save_adv_list(file_name, adv_list):
    """Saves generated list of adversarial images for adversarial training.

       :param file_name: Name of file to save to.
       :param adv_list: List of adversarial images.
    """

    path = verify_path(file_name, 'data/adversarial_lists/')
    torch.save(adv_list, path)


def save_adv_dataset(file_name, dataset):
    """Saves adversarial dataset for adversarial training.

       :param file_name: Name of file to save to.
       :param adv_list: List of adversarial images.
    """

    path = verify_path(file_name, 'data/adversarial_datasets/')
    torch.save(dataset, path)


def load_model(model, file_name):
    """Loads parameters of model.

       :param model: Model to load weights for.
       :param file_name: Name of file to load from.
    """

    path = verify_path(file_name, 'pretrained/')

    if not os.path.isfile(path):
        return model

    weights = torch.load(path)
    model.load_state_dict(weights)

    return model


def load_checkpoint(model, scheduler, optimizer, file_name):
    """Loads checkpoint from model training.

      :param model: Model to load weights for.
      :param scheduler: Scheduler to load state for.
      :param optimizer: Optimizer to load state for.
      :param file_name: Name of file to load from.
   """

    path = verify_path(file_name, 'checkpoints/')

    if not os.path.isfile(path):
        return model, scheduler, optimizer, 1, 0

    dct = torch.load(path)
    model.load_state_dict(dct['net'])
    scheduler.load_state_dict(dct['sch'])
    optimizer.load_state_dict(dct['opt'])
    epoch = dct['epoch']
    acc = dct['acc']

    return model, scheduler, optimizer, epoch, acc


def load_adv_list_checkpoint(file_name):
    """Loads adversarial training dataset checkpoint.

      :param file_name: Name of file to load from.
   """

    path = verify_path(file_name, 'checkpoints/')

    if not os.path.isfile(path):
        return

    dct = torch.load(path)
    index = dct['index']
    adv_list = dct['list']

    return adv_list, index


def load_adv_list(file_name):
    """Loads list of pre-generated adversarial images.

      :param file_name: Name of file to load from.
   """

    path = verify_path(file_name, 'data/adversarial_lists/')

    if not os.path.isfile(path):
        return

    adv_list = torch.load(path)
    return adv_list


def load_adv_dataset(file_name):
    """Loads adversarial dataset.

      :param file_name: Name of file to load from.
   """

    path = verify_path(file_name, 'data/adversarial_datasets/')

    if not os.path.isfile(path):
        return

    dataset = torch.load(path)
    return dataset
