from typing import Iterator

from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, DataLoader
from sat.util.store import save_model, save_checkpoint, load_checkpoint, \
    save_adv_list_checkpoint, load_adv_list_checkpoint, save_adv_list, save_adv_dataset
from sat.util.tqdm import get_tqdm, get_tqdm_iterable
import torch.nn as nn
import torch
from typing import Union
import math
import sat
import pickle


class AdversarialTraining:
    """Implementation of Adversarial Training"""

    def __init__(self,
                 model,
                 attacks,
                 args,
                 device='cpu',
                 norm=None,
                 transforms_train=None,
                 transforms_test=None):
        """

        :param model: The classifier to generate adversarial images for.
        :param attacks: List of adversarial attacks to defend against.
        :param args: Dict of keyword arguments for each attack in attack list.
        :param device: 'cuda' if running on gpu, 'cpu' otherwise.
        :param norm: Normalization transform function to use for normalizing image data.
        :param transforms_train: Transforms to use for training set. Should start with 'ToPILImage' function if
            using transforms requiring PIL image as input. Should not contain normalization transform.
        :param transforms_test: Transforms to use for test set. Should start with 'ToPILImage' function if
            using transforms requiring PIL image as input. Should not contain normalization transform.
        """

        self.attacks = attacks
        self.model = model
        self.device = device
        self.args = args
        self.norm = norm
        self.transforms_train = transforms_train
        self.transforms_test = transforms_test

    def train(self,
              classifier,
              trainloader,
              valloader,
              optimizer,
              criterion,
              scheduler,
              epochs=200,
              resume=False,
              model_name='advtrain'):
        """
        Trains a classifier on an adversarial dataset. The trained classifier will
        be saved to disk in 'pretrained' folder.

        :param classifier: The model to train.
        :param trainloader: Dataloader of trainset.
        :param valloader: Dataloader of validationset.
        :param optimizer: The optimizer to use during training.
        :param criterion: The loss function to use during training.
        :param scheduler: The learning rate scheduler to use during training.
        :param epochs: Number of epochs
        :param resume: If True, will continue from a checkpoint.
        :param model_name: Name of file to save classifier to.

        """

        total = epochs * (len(trainloader.dataset) + len(valloader.dataset))
        t = get_tqdm('Training', total, purge=False)
        best_acc = 0
        epoc = 1

        if resume:
            classifier, scheduler, optimizer, epoc, best_acc = load_checkpoint(classifier, scheduler,
                                                                               optimizer, model_name + '_ckpt.pth')
            t.update((epoc - 1) * (len(trainloader.dataset) + len(valloader.dataset)))

        for epoch in range(epoc, epochs + 1):
            sat.train(classifier, trainloader, criterion, optimizer, device=self.device, t=t, best_acc=best_acc)
            acc = sat.test(classifier, valloader, criterion, device=self.device, t=t, best_acc=best_acc)
            scheduler.step()

            if best_acc < acc:
                best_acc = acc
                save_checkpoint(classifier, scheduler, optimizer, epoch + 1, acc, model_name + '_ckpt.pth')

        save_model(classifier, model_name + '.pth')

    def test(self, classifier, testloader, should_generate=True, defence_list=None, batch_size=128, t_desc=None):
        """
        Tests accuracy of a classifier on an adversarial dataset. Can generate adversarial dataset from
        input dataset or use pre-generated list of adversarial images.

        :param classifier: Trained model to test.
        :param testloader: Dataloader object of original dataset when should_generate=True.
            List of adversarial images when should_generate=False.
        :param should_generate: If True, generates adversarial dataset from testloader.
            Set to False if using pre-generated dataset.
        :param defence_list: List of defence instances.
        :param batch_size: Batch size of adversarial image dataloader.
        :param t_desc: Optional custom description for tqdm.

        :return: Accuracy of test
        """

        if should_generate:
            adv_list = self.generate(testloader)
        else:
            adv_list = testloader

        data = [(adv.image_adv, adv.label_ori) for adv in adv_list]
        dataset = AdversarialTrainingDataset(data,
                                             device=self.device,
                                             transform=self.transforms_test,
                                             defence_list=defence_list)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        criterion = nn.CrossEntropyLoss()

        t = get_tqdm('Adversarial Testing' if t_desc is None else t_desc, len(dataloader.dataset), purge=True)
        acc = sat.test(classifier, dataloader, criterion, device=self.device, t=t)

        return acc

    def generate(self,
                 dataloader: torch.utils.data.DataLoader,
                 early_stopping: bool = True,
                 num_adv: Union[int, float] = None,
                 to_file: bool = False,
                 file_name: str = None,
                 resume: bool = False):
        """Generates list of adversarial images from a dataset.

        :param dataloader: Dataloader object for images to generate adversarials from.
        :param early_stopping: If True, will stop generating adversarial images once the enough have been generated.
        :param num_adv: The target number of adversarial images. Leave at None for all available images.
        :param to_file: If True, will save checkpoints to disk at regular intervals.
        :param file_name: Name of file to store checkpoints. Only relevant when to_file=True.
        :param resume: If True, will continue from last available checkpoint.

        :return: List of adversarial images
        """

        if early_stopping:
            assert num_adv is not None

        if resume:
            assert to_file

        if to_file:
            assert file_name is not None

        adv_list = list()

        if resume:
            resume_list, index = load_adv_list_checkpoint(file_name)

        for attack in self.attacks:
            att = attack(self.model, device=self.device, norm=self.norm)
            attack_name = att.__class__.__name__
            tmp_list = list()

            if resume:
                att._set_adv_list(resume_list)
                print('adv_list length: %s' % len(resume_list))
                print('att adv_list length: %s' % len(att.list))

            args = self.args[attack_name]
            print(args)
            t = get_tqdm_iterable(dataloader, 'Generate adversarial set', len(dataloader), purge=True)

            for i, data in t:

                if resume and i < index:
                    continue

                inputs, labels = data
                tmp_list = att(inputs, labels, **args)

                if to_file:
                    save_adv_list_checkpoint(file_name, tmp_list, i + 1)

                t.set_postfix({'attack': att.__class__.__name__, 'num_adversarials': len(tmp_list)})

                if early_stopping and len(tmp_list) > num_adv:
                    print('Enough images generated, stopping')
                    break

            adv_list = adv_list + tmp_list

        if to_file:
            save_adv_list(file_name, adv_list)

        return adv_list

    def make_loader(self,
                    adv_lists: list,
                    dataloader: torch.utils.data.DataLoader,
                    adv_ratio: float = 0.5,
                    val_ratio: float = 0.1,
                    fixed_dataset_size: bool = False,
                    to_file: bool = False,
                    dataset_name: str = None):
        """Generates list of adversarial images from a dataset.

        :param adv_lists: Sequence of different adversarial sample lists.
        :param dataloader: Dataloader object of dataset to merge with adversarial samples.
        :param to_file: If True, will save datasets to disk.
        :param dataset_name: Name of dataset. Only relevant when to_file=True.
        :param val_ratio: The ratio of images to use for validation set.
        :param adv_ratio: The ratio of adversarial images to non-adversarial images. Only relevant when
            fixed_dataset_size=True. Otherwise, all samples given will be used.
        :param fixed_dataset_size: If True, the dataset size will remain the same as the original dataset.
            To make this work, adversarial images will replace some images in original dataset.

        :return: Adversarial training and validation datasets
        """
        assert 1 >= adv_ratio > 0, "Error: adv_ratio must be between 1 and 0"
        assert 1 >= val_ratio >= 0, "Error: test_ratio must be between 1 and 0"

        num_adv = len(dataloader.dataset) * adv_ratio
        adv_list = []

        for lst in adv_lists:
            adv_list = adv_list + lst

        print('Number of original images: %s' % len(dataloader.dataset))
        print('Number of adversarial images: %s' % len(adv_list))

        if len(adv_list) < num_adv:
            num_adv = len(adv_list)

        num_adv = math.floor(num_adv)

        if fixed_dataset_size:
            indices_ori = torch.randperm(len(dataloader.dataset))[:len(dataloader.dataset) - num_adv]
            indices_adv = torch.randperm(len(adv_list))[:num_adv]
        else:
            indices_ori = range(len(dataloader.dataset))
            indices_adv = range(len(adv_list))

        arr_ori = [dataloader.dataset[i] for i in indices_ori]
        arr_adv = [(adv_list[i].image_adv, adv_list[i].label_ori) for i in indices_adv]

        num_test_ori = math.floor(len(arr_ori) * val_ratio)
        num_test_adv = math.floor(len(arr_adv) * val_ratio)

        train_arr_ori = arr_ori[:len(arr_ori) - num_test_ori]
        test_arr_ori = arr_ori[-num_test_ori:]

        train_arr_adv = arr_adv[:len(arr_adv) - num_test_adv]
        test_arr_adv = arr_adv[-num_test_adv:]

        print('Length of train ori arr: %s' % len(train_arr_ori))
        print('Length of train adv arr: %s' % len(train_arr_adv))
        print('Length of test ori arr: %s' % len(test_arr_ori))
        print('Length of test adv arr: %s' % len(test_arr_adv))

        train_dataset = AdversarialTrainingDataset(train_arr_ori + train_arr_adv,
                                                   self.device, transform=self.transforms_train)
        print('Length of training dataset: %s' % len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

        test_dataset = AdversarialTrainingDataset(test_arr_ori + test_arr_adv,
                                                  self.device, transform=self.transforms_test)
        print('Length of validation dataset: %s' % len(test_dataset))
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0)

        self.trainloader = train_loader
        self.testloader = test_loader

        if to_file:
            save_adv_dataset(dataset_name + '_train.pth', train_loader)
            save_adv_dataset(dataset_name + '_val.pth', test_loader)

        return train_loader, test_loader


class AdversarialTrainingDataset(Dataset):
    """
    Dataset used for adversarial training.
    """
    def __init__(self, data, device, transform=None, defence_list=None):
        """

        :param data: The adversarial image data. Assumed to be a list of 'AdversarialSample' objects.
        :param device: 'cuda' if running on gpu, 'cpu' otherwise.
        :param transform: Any image transforms to apply to data.
        :param defence_list: List of defence instances.
        """
        super(AdversarialTrainingDataset, self).__init__()

        self.data = data
        self.n_samples = len(self.data)
        self.device = device
        self.transform = transform
        self.defences = defence_list

    def __getitem__(self, index) -> T_co:
        """

        :param index: Index of sample to fetch
        :return: Tuple of (image, target) for sample
        """
        data = self.data[index]
        sample = data[0]

        if self.defences is not None:
            for defence in self.defences:
                sample = defence(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        target = data[1]

        return sample, target

    def __len__(self):
        """

        :return: Number of samples in dataset
        """
        return self.n_samples
