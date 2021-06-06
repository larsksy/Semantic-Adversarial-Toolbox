from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co
import torch


class ProcessingDefence:
    """Common class for data augmentation defences."""

    def __init__(self):
        pass

    def apply(self, dataloader, shuffle=True):
        """Applies defence to a dataloader object.

        :param dataloader: The Dataloader object to apply the defence to.
        :param shuffle: If True, will shuffle the data in the dataloader.
        """
        samples = list()
        targets = list()
        num_workers = dataloader.num_workers
        batch_size = dataloader.batch_size

        for i, (data, labels) in enumerate(dataloader):
            samples.append(data)
            targets.append(labels)

        samples = torch.cat(samples, dim=0)
        targets = torch.cat(targets, dim=0)
        new_dataset = DefenceDataset(samples, targets, self)
        new_dataloader = DataLoader(new_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return new_dataloader


class DefenceDataset(Dataset):
    """Custom dataset used for applying a defence to a dataloader. """

    def __init__(self, samples, targets, defence):
        """

        :param samples: Tensor of data samples.
        :param targets: Tensor of data labels.
        :param defence: The defence to apply.
        """
        super(DefenceDataset, self).__init__()

        self.samples = samples
        self.targets = targets
        self.n_samples = len(self.samples)
        self.defence = defence

    def __getitem__(self, index) -> T_co:
        """

        :param index: Index of sample to fetch
        :return: Tuple of (image, target) for sample
        """
        sample = self.samples[index]

        if self.defence is not None:
            sample = self.defence(sample)

        target = self.targets[index]

        return sample, target

    def __len__(self):
        """

        :return: Number of samples in dataset
        """
        return self.n_samples
