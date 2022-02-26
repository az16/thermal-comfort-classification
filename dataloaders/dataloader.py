from torch.utils.data.dataset import Dataset
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, split):
        self.split = split
        if 'train' in split:
            self.transform = self.training_preprocess
        elif split == 'val':
            self.transform = self.validation_preprocess
        elif split == 'test':
            self.transform = self.test_preprocess
        else:
            raise (RuntimeError("Invalid dataset type: " + split + "\nSupported dataset types are: train, val, test"))

    def training_preprocess(self, rgb, depth):
        raise NotImplementedError()

    def validation_preprocess(self, rgb, depth):
        raise NotImplementedError()

    def test_preprocess(self, rgb, depth):
        raise NotImplementedError()

    def get_raw(self, index):
        raise NotImplementedError()

    def __getitem__(self, index):
        rgb, depth = self.get_raw(index)
        return self.transform(rgb, depth)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def add_dataset_specific_args(parser):
        parser.add_argument('--path', required=True, type=str, help='Path to dataset')
        parser.add_argument('--training',   action="store_true", help='dataset for training')
        parser.add_argument('--validation', action="store_true", help='dataset for validation')
        parser.add_argument('--test',       action="store_true", help='dataset for test')

class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.transform = None
        self.datasets = datasets
        self.indices = np.hstack([[dataset_index] * len(d) for dataset_index, d in enumerate(self.datasets)])
        np.random.shuffle(self.indices)

    def __getitem__(self, i):
        if not self.transform is None:
            for dataset in self.datasets:
                dataset.transform = lambda x,y: (x,y)
        item_index = (self.indices[0:i] == self.indices[i]).sum()
        rgb, depth = self.datasets[self.indices[i]][item_index]
        if self.transform is None:
            return rgb, depth
        else:
            return self.transform(rgb, depth)

    def __len__(self):
        return sum(len(d) for d in self.datasets)