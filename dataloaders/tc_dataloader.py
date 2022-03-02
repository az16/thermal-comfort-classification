from dataset import *
from path import Path
from tqdm import tqdm
from dataloaders.utils import *

import torch
import numpy as np
import os
import csv

class TC_Dataloader(BaseDataset):
    def __init__(self, root, split):
        self.split = split 
        self.root = root 
        self.load_file_contents(split)
        if self.split == "training": self.preprocess = self.training_preprocess
        elif self.split == "validation": self.preprocess = self.validation_preprocess
        print("Starting pre-processing..")
        self.preprocess()    
    
    def load_file_contents(self, root, split):
        print("Searching for files..")
        file_names = os.listdir(Path.db_root_dir("tcs"))
        assert len(file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
        
        if split == "training": file_names=file_names[0:15]
        elif split == "validation": file_names = file_names[16:19]
        
        tmp = []
        print("Loading contents..")
        with open(file_names, "r") as f:
            reader = list(csv.reader(f, delimiter=';'))
            reader.pop(0) #get rid of header
            tmp.extend(reader)
            
        tmp = np.array(tmp)
        self.data = tmp[:,1:26]
        self.labels = tmp[:,26]
        print("File contents loaded!")
        
    def training_preprocess(self):
        
        #set up individual data signals with appropriate type
        self.data_frame = {"age" : self.data[:,0].astype(np.float32),
                            "gender" : one_hot(self.data[:,1]),
                            "pmv_index" : self.data[:,2:5].astype(np.float32),
                            "rgb_path" : self.data[:,6],
                            "key_points" : to_keypoint(self.data[:,7:20]),
                            "heart_rate" : self.data[:,20].astype(np.float32),
                            "wrist_temp" : self.data[:,21].astype(np.float32),
                            "t_a" : self.data[:,22].astype(np.float32),
                            "rh_a" : self.data[:,23].astype(np.float32)/100,
                            "emotion" : one_hot(self.data[:,24], classes=6)}
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
        self.data_frame["age"] = norm(self.data_frame["age"], min=0, max=100)
        self.data_frame["heart_rate"] = norm(self.data_frame["heart_rate"], min=40, max=100)
        self.data_frame["wrist_temp"] = norm(self.data_frame["wrist_temp"])
        self.data_frame["t_a"] = norm(self.data_frame["t_a"])
        self.data_frame["rh_a"] = norm(self.data_frame["rh_a"])
        
        print("Preprocessing done!")
        
        
    def validation_preprocess(self):
        #set up individual data signals with appropriate type
        self.data_frame = {"age" : self.data[:,0].astype(np.float32),
                            "gender" : one_hot(self.data[:,1]),
                            "pmv_index" : self.data[:,2:5].astype(np.float32),
                            "rgb_path" : self.data[:,6],
                            "key_points" : to_keypoint(self.data[:,7:20]),
                            "heart_rate" : self.data[:,20].astype(np.float32),
                            "wrist_temp" : self.data[:,21].astype(np.float32),
                            "t_a" : self.data[:,22].astype(np.float32),
                            "rh_a" : self.data[:,23].astype(np.float32)/100,
                            "emotion" : one_hot(self.data[:,24], classes=6)}
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
        self.data_frame["age"] = norm(self.data_frame["age"], min=0, max=100)
        self.data_frame["heart_rate"] = norm(self.data_frame["heart_rate"], min=40, max=100)
        self.data_frame["wrist_temp"] = norm(self.data_frame["wrist_temp"])
        self.data_frame["t_a"] = norm(self.data_frame["t_a"])
        self.data_frame["rh_a"] = norm(self.data_frame["rh_a"])
    
    def __getitem__(self, index):
        rgb = to_tensor((rgb_loader(self.root, self.data_frame["rgb_path"][index])))/255
        key_points = torch.from_numpy(self.data_frame["key_points"][index])
        ratio_scaled = [self.data_frame["age"][index],
                        self.data_frame["pmv_index"][index],
                        self.data_frame["heart_rate"][index],
                        self.data_frame["wrist_temp"][index],
                        self.data_frame["t_a"][index],
                        self.data_frame["rh_a"][index]]
        one_hot_encoded = [self.data_frame["gender"][index],
                           self.data_frame["emotion"][index]]
        label = self.labels[index]
        
        return (ratio_scaled, one_hot_encoded, key_points, rgb), label
    
    def __len__(self):
        return len(self.data.shape[0])
        
