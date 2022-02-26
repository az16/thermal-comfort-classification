from torch.utils.data.dataset import Dataset

class BaseDataset(Dataset):
    def __init__(self, split):
        self.split = split
        if split == 'train':
            self.transform = self.training_preprocess
        elif split == 'val':
            self.transform = self.validation_preprocess
        else:
            raise (RuntimeError("Invalid dataset type: " + split + "\nSupported dataset types are: train, val"))

    def training_preprocess(self, rgb, depth):
        raise NotImplementedError()

    def validation_preprocess(self, rgb, depth):
        raise NotImplementedError()

    def get_raw(self, index):
        raise NotImplementedError()

    def __getitem__(self, index):
        rgb, depth = self.get_raw(index)
        return self.transform(rgb, depth)

    def __len__(self):
        return len(self.images)