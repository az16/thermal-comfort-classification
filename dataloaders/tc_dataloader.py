from msilib import sequence
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
# 3         weight
# 4         height
# 5         bodyfat
# 6         bodytemp
# 7         sport
# 8         meal 
# 9         tiredness
# 10        black-globe-temp
# 11        pce-ambient
# 12        clothing-level
# 13        air-velocity
# 14        metabolic rate
# 15        emotion
# 16        emotion-ml
# 17        rgb paths
# 18-29     key-points
# 30,33     bio-signals
# 34,35     environmental-signals
# 36        label

#idea:
#load in all lines
#split into 80% train - 20% val

class TC_Dataloader(BaseDataset):
    """
    Loads .csv data and preprocesses respective splits

    Args:
        BaseDataset (Dataset): loads and splits dataset
    """
    def __init__(self, root, split, preprocess, use_sequence=False, sequence_size=10, use_imgs=False, output_size=(228, 228), data_augmentation=True, use_demographic=False, use_pmv_vars=True, use_physiological=True):
        self.split = split 
        self.root = root 
        self.preprocessing_config = preprocess #bool or dict of bools that define which signals to preprocess
        self.augment_data = data_augmentation
        self.output_size = output_size
        self.use_sequence = use_sequence
        self.sequence_size = sequence_size 
        self.use_demographic = use_demographic
        self.use_imgs = use_imgs
        self.use_pmv_vars = use_pmv_vars
        self.use_physiological = use_physiological
        self.load_file_contents(split)
        
        self.demographic_keys = ["age", "gender", "weight", "height", "bodyfat", "bodytemp", "sport", "meal", "tiredness"]
        self.physiological_keys = ["heart_rate", "wrist_temp", "gsr"]
        self.pmv_keys = ["pce_ta", "pce_tg", "rh", "clothing"]
        
        #do dataaugmentation in case images are used
        if self.use_imgs:
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
        split_limit = int(len(file_names)*0.8)
        if split == "training": file_names=file_names[:split_limit]
        elif split == "validation": file_names = file_names[split_limit:]
        
        #load .csv contents as list
        tmp = []
        print("Loading contents..")
        with open(file_names, "r") as f:
            reader = list(csv.reader(f, delimiter=';'))
            reader.pop(0) #get rid of header
            tmp.extend(reader)
        
        #turn into np array and split input from labels
        tmp = np.array(tmp)
        self.data = tmp[:,1:]
        #self.labels = tmp[:,33]
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
                            "weight":self.data[:,2].astype(np.float32),
                            "height":self.data[:,3].astype(np.float32),
                            "bodyfat:":self.data[:,4].astype(np.float32),
                            "bodytemp":self.data[:,5].astype(np.float32),
                            "sport":self.data[:,6],
                            "meal":self.data[:,7],
                            "tiredness":self.data[:,8],
                            "clothing": self.data[:,9],
                            "pce_tg":self.data[:,10].astype(np.float32),
                            "pce_ta":self.data[:,11].astype(np.float32),
                            "rgb_path" : self.data[:,16],
                            "key_points" : to_keypoint(self.data[:,17:27]),
                            "heart_rate" : self.data[:,28].astype(np.float32),
                            "wrist_temp" : self.data[:,29].astype(np.float32),
                            "gsr" : self.data[:,30].astype(np.float32),
                            "t_a" : self.data[:,31].astype(np.float32),
                            "rh_a" : self.data[:,32].astype(np.float32)/100,
                            "emotion_ml" : self.data[:,15],
                            "emotion" : self.data[:,14],
                            "labels" : self.data[:,33].astype(np.float32)}
        #self.labels = self.labels.astype(np.float32)
        
        #find outliers and missing values
        masks = []
        if self.use_physiological:
            mask_hr = clean(self.data_frame["heart_rate"])
            mask_wst = clean(self.data_frame["wrist_temp"])
            mask_gsr = clean(self.data_frame["gsr"])
            masks.append(make_mask([mask_hr, mask_wst, mask_gsr]))
        
        if self.use_pmv_vars:
            mask_ta = clean(self.data_frame["t_a"])
            mask_rh = clean(self.data_frame["rh_a"])
            mask_pce_ta = clean(self.data_frame["pce_ta"])
            mask_pce_tg = clean(self.data_frame["pce_tg"])
            masks.append(make_mask([mask_ta, mask_rh, mask_pce_ta, mask_pce_tg])) 
        
        if len(masks) > 0:
            full_mask = make_mask(masks)
        
            for key in self.data_frame:
                val = self.data_frame[key]
                self.data_frame.update({key: val[full_mask]})
        
        #calculate pmv index
        #self.data_frame["pmv_index"] = pmv(self.data_frame["pmv_index"], self.data_frame["t_a"], self.data_frame["rh_a"])
        
        #normalize where necessary
        if isinstance(self.preprocessing_config, bool) and self.preprocessing_config:
            self.data_frame["age"] = norm(self.data_frame["age"], min=0, max=100)
            self.data_frame["weight"] = norm(self.data_frame["weight"])
            self.data_frame["height"] = norm(self.data_frame["height"])
            self.data_frame["bodyfat"] = norm(self.data_frame["bodyfat"])
            self.data_frame["bodytemp"] = norm(self.data_frame["bodytemp"], max=42.0)
            self.data_frame["sport"] = (self.data_frame["sport"] == 1)
            self.data_frame["meal"] = norm(self.data_frame["meal"])
            self.data_frame["tiredness"] = norm(self.data_frame["tiredness"], min=0, max=10)
            self.data_frame.update({"heart_rate": norm(self.data_frame["heart_rate"], min=40, max=120)})
            self.data_frame.update({"wrist_temp" : norm(self.data_frame["wrist_temp"])})
            self.data_frame.update({"t_a" : norm(self.data_frame["t_a"])})
            self.data_frame.update({"gender": one_hot(self.data_frame["gender"], classes=3)})
            self.data_frame.update({"emotion" : one_hot(self.data_frame["emotion"], classes=7)})
            self.data_frame.update({"emotion-ml" : one_hot(self.data_frame["emotion-ml"], classes=6)})
        else:
            for key in ["age", "heart_rate", "wrist_temp", "t_a", "rh_a", "gender", "emotion", "emotion_ml"]:
                if self.preprocessing_config[key]:
                    if key == "age": self.data_frame["age"] = norm(self.data_frame["age"], min=0, max=100)
                    elif key == "weight": self.data_frame["weight"] = norm(self.data_frame["weight"])
                    elif key == "height": self.data_frame["height"] = norm(self.data_frame["height"])
                    elif key == "bodyfat": self.data_frame["bodyfat"] = norm(self.data_frame["bodyfat"])
                    elif key == "bodytemp": self.data_frame["bodytemp"] = norm(self.data_frame["bodytemp"], max=42.0)
                    elif key == "sport": self.data_frame["sport"] = (self.data_frame["sport"] == 1)
                    elif key == "meal": self.data_frame["meal"] = norm(self.data_frame["meal"])
                    elif key == "tiredness": self.data_frame["tiredness"] = norm(self.data_frame["tiredness"], min=0, max=10)
                    elif  key == "heart_rate": self.data_frame.update({"heart_rate": norm(self.data_frame["heart_rate"], min=40, max=120)})
                    elif key == "wrist_temp": self.data_frame.update({"wrist_temp" : norm(self.data_frame["wrist_temp"])})
                    elif key == "t_a": self.data_frame.update({"t_a" : norm(self.data_frame["t_a"])})
                    elif key == "gender": self.data_frame.update({"gender": one_hot(self.data_frame["gender"], classes=3)})
                    elif key == "emotion": self.data_frame.update({"emotion" : one_hot(self.data_frame["emotion"], classes=7)})
                    elif key == "emotion-ml": self.data_frame.update({"emotion-ml" : one_hot(self.data_frame["emotion-ml"], classes=6)})
        
        print("Preprocessing done!")
    
    def train_transform(self, rgb):
        if not rgb is None:
            angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
            do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

            rgb = rotate(angle, rgb)
            rgb = center_crop(rgb, self.output_size)
            rgb = horizontal_flip(rgb, do_flip)
            rgb = np.asfarray(rgb, dtype='float') / 255

        #normed, keypoints = augmentation((normed, keypoints))
        
       
        #keypoints = np.array([[x[0]/1920, x[1]/1080], x[2]/5000] for x in keypoints)
        
        return rgb
    
    def val_transform(self, rgb):
        #normed, keypoints, rgb = features

        rgb = center_crop(rgb, self.output_size)
        rgb = np.asfarray(rgb, dtype='float') / 255

        #normed, keypoints = augmentation((normed, keypoints), val=True)
        
        #keypoints = np.array([[x[0]/1920, x[1]/1080], x[2]/8000] for x in keypoints)
        
        return rgb
    
    def __getitem__(self, index):
        """
        Creates input,label pair with data found in each data frame at
        the given index.

        Args:
            index (int): the index to get the data from

        Returns:
            tupel(torch.tensor, torch.tensor, torch.tensor), float: input and labels at index
        """
        # key_points = self.data_frame["key_points"][index]
        # normed = np.array([self.data_frame["age"][index],
        #                 self.data_frame["pmv_index"][index],
        #                 self.data_frame["heart_rate"][index],
        #                 self.data_frame["wrist_temp"][index],
        #                 self.data_frame["t_a"][index],
        #                 self.data_frame["rh_a"][index]])
        # one_hot_encoded = (torch.from_numpy(self.data_frame["gender"][index]),
        #                    torch.from_numpy(self.data_frame["emotion"][index]),
        #                    torch.from_numpy(self.data_frame["emotion-ml"][index]))
    
        # label = self.labels[index]
        
        # normed, key_points, rgb = self.transform(normed, key_points, rgb)
        
        
        # return (torch.from_numpy(normed), one_hot_encoded, torch.from_numpy(key_points), to_tensor(rgb)), label
        limit = index+1
        if self.use_sequence:
            limit = index+self.sequence_size+1
        
        out = []
        
        if self.use_demographic:
            out.extend([torch.from_numpy(self.data_frame[x][index:limit]) for x in self.demographic_keys])
        
        if self.use_imgs:
            out.append(self.transform(rgb_loader(self.root, self.data_frame["rgb_path"][index:limit])))
        
        if self.use_physiological:
            out.extend([torch.from_numpy(self.data_frame[x][index:limit]) for x in self.physiological_keys])
        
        if self.use_pmv_vars:
            out.extend([torch.from_numpy(self.data_frame[x][index:limit]) for x in self.pmv_keys])
            
        return torch.cat(out, dim=2)
    
    def __len__(self):
        """
        Returns the number of total data lines. Columns
        are not returned as they are the same for both, training
        and validation dataset.

        Returns:
            int: number of rows int the dataset
        """        
        return len(self.data.shape[0])
        
