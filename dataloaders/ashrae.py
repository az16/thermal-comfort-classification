from torch.utils.data.dataset import Dataset
from pathlib import Path
import csv
from enum import Enum
import torch
import numpy as np
from dataloaders.utils import label2idx, order_representation

class HEADER(Enum):
    PUBLICATION = 0
    CONTRIBUTOR = 1
    YEAR        = 2
    SEASON      = 3
    KOPPEN_CLIMATE_CLASS = 4
    CLIMATE              = 5
    CITY                 = 6
    COUNTRY              = 7
    BUILDING_TYPE        = 8
    COOLING_STRAT        = 9
    COOLING_STRAT_OP     = 10
    HEATING_STRAT        = 11
    AGE                  = 12
    SEX                  = 13
    THERMAL_SENSATION    = 14
    THERMAL_ACCEPTABILITY = 15
    THERMAL_PREFERENCE    = 16
    AIR_MOVEMENT_ACCEPT   = 17
    AIR_MOVEMENT_PREF     = 18
    THERMAL_COMFORT       = 19
    PMV                   = 20
    PPD                   = 21
    SET                   = 22
    CLO                   = 23
    MET                   = 24
    ACTIVITY_10           = 25
    ACTIVITY_20           = 26
    ACTIVITY_30           = 27
    ACTIVITY_60           = 28
    AIR_TEMPERATURE_C     = 29
    AIR_TEMPERATURE_F     = 30
    TA_H_C                = 31
    TA_H_F                = 32
    TA_M_C                = 33
    TA_M_F                = 34
    TA_L_C                = 35
    TA_L_F                = 36
    OPERATIVE_TEMP_C      = 37
    OPERATIVE_TEMP_F      = 38
    RADIANT_TEMPERATURE_C = 39
    RADIANT_TEMPERATURE_F = 40
    GLOBE_TEMPERATURE_C   = 41
    GLOBE_TEMPERATURE_F   = 42
    TG_H_C                = 43
    TG_H_F                = 44
    TG_M_C                = 45
    TG_M_F                = 46
    TG_L_C                = 47
    TG_L_F                = 48
    RELATIVE_HUMIDITY     = 49
    HUMIDITY_PREFERENCE   = 50
    HUMIDITY_SENSATION    = 51

N = 31505
N_TRAIN = int(N // 1.25)
N_VALID = N - N_TRAIN

class ASHRAE_Dataset():
    def __init__(self, path, cols, scale, split):
        self.split = split
        self.path = Path(path)
        self.cols = [HEADER(c) for c in cols]
        self.scale = scale
        self.load_csv()
        self.preprocess()
        print("Found {} entries for split {}.".format(len(self.data), self.split))

    def splits(self):
        return self.data[0:N_TRAIN, 0:-1], self.data[0:N_TRAIN, -1], self.data[N_TRAIN:, 0:-1], self.data[N_TRAIN:, -1], None, None

    def load_csv(self):
        self.data = []
        with open(self.path/"ashrae_db2.01.csv", newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i, row in enumerate(spamreader):
                row = [row[col.value] for col in self.cols]
                if i == 0: 
                    print("Using ", row)
                    continue
                if "NA" in row: continue
                self.data.append(row)
        self.data = np.asarray(self.data).astype(np.float32)

    def preprocess(self):
        minimum = np.min(self.data, axis=0)
        maximum = np.max(self.data, axis=0)
        self.data[:, 0:-1] = (self.data[:, 0:-1] - minimum[0:-1]) / (maximum[0:-1] - minimum[0:-1])

        print(np.histogram(np.rint(self.data[:, -1]), bins=[-3, -2, -1, 0, 1, 2, 3]))
        if self.split == "training":
            self.data = self.data[0:N_TRAIN]
        elif self.split == "validation":
            self.data = self.data[N_TRAIN:]
        else:
            self.data = self.data


class ASHRAE_Dataloader(ASHRAE_Dataset):
    def __init__(self, path, cols, scale, split, data_augmentation=False, *args, **kwargs) -> None:
        super().__init__(path, cols, scale, split)
        self.data_augmentation = data_augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dp = self.data[index]
        X = torch.from_numpy(dp[0:-1]).unsqueeze(0)
        label = np.rint(dp[-1])

        label = label2idx(label, scale=self.scale)
        label = order_representation(label, scale=self.scale)
        Y = torch.from_numpy(label)

        if self.data_augmentation and np.random.rand() > 0.5:
            X += torch.randn_like(X) * 0.02

        return X, Y


if __name__ == '__main__':
    #dataset = ASHRAE_Dataloader(path="H:/data/ASHRAE", cols=[HEADER.RADIANT_TEMPERATURE_C, HEADER.AIR_TEMPERATURE_C, HEADER.RELATIVE_HUMIDITY, HEADER.THERMAL_SENSATION], scale=7, split="training")
    dataset = ASHRAE_Dataset(path="H:/data/ASHRAE", cols=[HEADER.RADIANT_TEMPERATURE_C, HEADER.AIR_TEMPERATURE_C, HEADER.RELATIVE_HUMIDITY, HEADER.THERMAL_SENSATION], scale=7, split="training")
