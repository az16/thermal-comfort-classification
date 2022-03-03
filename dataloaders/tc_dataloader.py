from dataset import *
from path import Path
from tqdm import tqdm
from dataloaders.utils import *

import torch
import numpy as np
import os
import csv

# raw data table
# index     name       
# 0         timestamp      
# 1         age       
# 2         gender      
# 3         black-globe-temp
# 4         clothing-level
# 5         air-velocity
# 6         metabolic rate
# 7         rgb paths
# 8-19      key-points
# 20,21     bio-signals
# 22,23     environmental-signals
# 24        emotion
# 25        label


class TC_Dataloader(BaseDataset):
    """
    Loads .csv data and preprocesses respective splits

    Args:
        BaseDataset (Dataset): loads and splits dataset
    """
    def __init__(self, root, split, preprocess, output_size, data_augmentation=True):
        self.split = split 
        self.root = root 
        self.preprocessing_config = preprocess
        self.augment_data = data_augmentation
        self.output_size = output_size
        self.load_file_contents(split)
        
        if self.split == "training": self.transform = self.train_transform
        elif self.split == "validation": self.transform = self.val_transform
        
        if not self.preprocessing_config is None:
            print("Starting pre-processing..")
            self.preprocess()    
    
    def load_file_contents(self, root, split):
        """
        Loads .csv data as np.array and splits input signals/labels.


        Args:
            root (str): path of the root directory
            split (str): training or validation split string
        """
        
        #find files
        print("Searching for files..")
        file_names = os.listdir(Path.db_root_dir("tcs"))
        assert len(file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
        
        #split
        if split == "training": file_names=file_names[0:15]
        elif split == "validation": file_names = file_names[16:19]
        
        #load .csv contents as list
        tmp = []
        print("Loading contents..")
        with open(file_names, "r") as f:
            reader = list(csv.reader(f, delimiter=';'))
            reader.pop(0) #get rid of header
            tmp.extend(reader)
        
        #turn into np array and split input from labels
        tmp = np.array(tmp)
        self.data = tmp[:,1:25]
        self.labels = tmp[:,25]
        print("File contents loaded!")
        
    def preprocess(self):
        """
        Function that assigns appropriate type for each input modality and
        performs preprocessing steps like data cleaning, one-hot-encoding
        and normalization to [0,1]
        """
        #set up individual data signals with appropriate type
        self.data_frame = {"age" : self.data[:,0].astype(np.float32),
                            "gender" : self.data[:,1],
                            "pmv_index" : self.data[:,2:5].astype(np.float32),
                            "rgb_path" : self.data[:,6],
                            "key_points" : to_keypoint(self.data[:,7:20]),
                            "heart_rate" : self.data[:,20].astype(np.float32),
                            "wrist_temp" : self.data[:,21].astype(np.float32),
                            "t_a" : self.data[:,22].astype(np.float32),
                            "rh_a" : self.data[:,23].astype(np.float32)/100,
                            "emotion" : self.data[:,24]}
        self.labels = self.labels.astype(np.float32)
        
        #find outliers and missing values
        mask_hr = clean(self.data_frame["heart_rate"])
        mask_wst = clean(self.data_frame["wrist_temp"])
        mask_ta = clean(self.data_frame["t_a"])
        mask_rh = clean(self.data_frame["rh_a"])
        
        full_mask = np.logical_or(np.logical_or(np.logical_or(mask_hr, mask_wst), mask_ta), mask_rh)
        
        for key in self.data_frame:
            val = self.data_frame[key]
            self.data_frame.update({key: val[full_mask]})
        
        #calculate pmv index
        self.data_frame["pmv_index"] = pmv(self.data_frame["pmv_index"], self.data_frame["t_a"], self.data_frame["rh_a"])
        
        #normalize where necessary
        if isinstance(self.preprocessing_config, bool) and self.preprocessing_config:
            self.data_frame["age"] = norm(self.data_frame["age"], min=0, max=100)
            self.data_frame["heart_rate"] = norm(self.data_frame["heart_rate"], min=40, max=120)
            self.data_frame["wrist_temp"] = norm(self.data_frame["wrist_temp"])
            self.data_frame["t_a"] = norm(self.data_frame["t_a"])
            one_hot(self.data_frame["gender"], classes=3)
            one_hot(self.data_frame["emotion"], classes=6)
        else:
            for key in ["age", "heart_rate", "wrist_temp", "t_a", "rh_a", "gender", "emotion"]:
                if self.preprocessing_config[key] and key == "age": self.data_frame["age"] = norm(self.data_frame["age"], min=0, max=100)
                elif self.preprocessing_config[key] and key == "heart_rate": self.data_frame["heart_rate"] = norm(self.data_frame["heart_rate"], min=40, max=120)
                elif self.preprocessing_config[key] and key == "wrist_temp": self.data_frame["wrist_temp"] = norm(self.data_frame["wrist_temp"])
                elif self.preprocessing_config[key] and key == "t_a": self.data_frame["t_a"] = norm(self.data_frame["t_a"])
                elif self.preprocessing_config[key] and key == "gender": one_hot(self.data_frame["gender"], classes=3)
                elif self.preprocessing_config[key] and key == "emotion": one_hot(self.data_frame["emotion"], classes=6)
        
        print("Preprocessing done!")
    
    def train_transform(self, features):
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        normed, keypoints, rgb = features

        rgb = rotate(angle, rgb)
        rgb = center_crop(rgb, self.output_size)
        rgb = horizontal_flip(rgb, do_flip)
        rgb = np.asfarray(rgb, dtype='float') / 255

        normed, keypoints = augmentation((normed, keypoints))
        
        keypoints = np.array([[x[0]/1920, x[1]/1080], x[2]/8000] for x in keypoints)
        
        return normed, keypoints, rgb
    
    def val_transform(self, features):
        normed, keypoints, rgb = features

        rgb = center_crop(rgb, self.output_size)
        rgb = np.asfarray(rgb, dtype='float') / 255

        normed, keypoints = augmentation((normed, keypoints), val=True)
        
        keypoints = np.array([[x[0]/1920, x[1]/1080], x[2]/8000] for x in keypoints)
        
        return normed, keypoints, rgb
    
    def __getitem__(self, index):
        """
        Creates input,label pair with data found in each data frame at
        the given index.

        Args:
            index (int): the index to get the data from

        Returns:
            tupel(torch.tensor, torch.tensor, torch.tensor), float: input and labels at index
        """
        rgb = rgb_loader(self.root, self.data_frame["rgb_path"][index])
        key_points = self.data_frame["key_points"][index]
        normed = np.array([self.data_frame["age"][index],
                        self.data_frame["pmv_index"][index],
                        self.data_frame["heart_rate"][index],
                        self.data_frame["wrist_temp"][index],
                        self.data_frame["t_a"][index],
                        self.data_frame["rh_a"][index]])
        one_hot_encoded = (torch.from_numpy(self.data_frame["gender"][index]),
                           torch.from_numpy(self.data_frame["emotion"][index]))
        label = self.labels[index]
        
        normed, key_points, rgb = self.transform(normed, key_points, rgb)
        
        
        return (torch.from_numpy(normed), one_hot_encoded, torch.from_numpy(key_points), to_tensor(rgb)), label
    
    def __len__(self):
        """
        Returns the number of total data lines. Columns
        are not returned as they are the same for both, training
        and validation dataset.

        Returns:
            int: number of rows int the dataset
        """        
        return len(self.data.shape[0])
        
