import multiprocessing
from dataloaders.dataset import *
from dataloaders.path import Path
from tqdm import tqdm
from dataloaders.utils import *
from dataloaders.path import Path
from multiprocessing import Process, Queue
import pandas as pd 

import torch
import numpy as np
import os

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
# 17-26     key-points
# 27,29     bio-signals
# 30,31     environmental-signals
# 32        label


class TC_Dataloader(BaseDataset):
    """
    Loads .csv data and preprocesses respective splits

    Args:
        BaseDataset (Dataset): loads and splits dataset
    """
    def __init__(self, root, split, preprocess=None, use_sequence=False, sequence_size=10, output_size=(224, 224), continuous_labels=False, data_augmentation=True, cols=None, image_path=None, use_imgs=False):
        self.split = split 
        self.root = root 
        self.preprocessing_config = preprocess #bool or dict of bools that define which signals to preprocess
        self.augment_data = data_augmentation
        self.output_size = output_size
        self.use_imgs = use_imgs
        self.use_sequence = use_sequence
        self.sequence_size = sequence_size 
        self.columns = [header[x] for x in cols]
        self.continuous_labels = continuous_labels
        assert not cols is None, "Specify which columns to use as inputs for training."
        print("Using these features: {0}".format(self.columns))
        self.rgb_transform = None 
        self.img_list = None 
        if self.use_imgs:
            if self.split == "training": self.rgb_transform = self.train_transform
            else: self.rgb_transform = self.val_transform
        self.load_file_contents(split)
        
        
        #do dataaugmentation in case images are used
        if self.augment_data and "RGB_Fronal_View" in cols:
            if self.split == "training": self.transform = self.train_transform
            elif self.split == "validation": self.transform = self.val_transform
        
        #if not self.preprocessing_config is None:
        self.preprocess()    
    
    def load_file_contents(self, split):
        """
        Loads .csv data as np.array and splits input signals/labels.


        Args:
            root (str): path of the root directory
            split (str): training or validation split string
        """
        
        #find files
        print("Searching for {0} files..".format(self.split))
        file_names = [] #os.listdir(Path.db_root_dir("tcs"))
        with open("./dataloaders/splits/{0}_{1}.txt".format(self.split, 60)) as file:
            lines = file.readlines()
            file_names = [line.rstrip() for line in lines]
        assert len(file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
            
        file_names = [Path.db_root_dir("tcs")+x for x in file_names]
        print("Found {0} {1} files at {2}".format(len(file_names),self.split,Path.db_root_dir("tcs")))
        
        # train_limit = int(len(file_names)*0.6)
        # val_size = int((len(file_names)-train_limit)*0.5)
        # val_limit = train_limit+val_size
        # test_size = int(len(file_names)-val_limit)
        # if split == "training": file_names = file_names[:train_limit]; print("Using {0} files for {1}".format(train_limit, self.split))
        # elif split == "validation": file_names = file_names[train_limit:val_limit]; print("Using {0} files for {1}".format(val_size, self.split))
        # elif split == "test": file_names = file_names[val_limit:]; print("Using {0} files for {1}".format(test_size, self.split))

        #load .csv contents as list
        print("Loading contents..")
        self.df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(file_names)])
        print("Creating data frames..")
        
        # limit = 0
        # size = len(self.df)
        # if split == "training":
        #     limit = int(size*0.6)
        #     self.df = self.df[0:limit]
        # elif split == "validation":
        #     limit = int(size*0.6), int(size*0.6)+int(size*0.2)
        #     self.df = self.df[limit[0]:limit[1]]
        # elif split == "test":
        #     limit = size - int(size*0.2)
        #     self.df = self.df[limit:]
        print("File contents loaded!")
        
    def preprocess(self):
        """
        Function that assigns appropriate type for each input modality and
        performs preprocessing steps like data cleaning, one-hot-encoding
        and normalization to [0,1]
        """
        if self.preprocessing_config is None: return
        print("Pre-processing..")
        masks = []
        if not self.columns == image_only:
            #Load types
            data_type_dict = dict({})
            for key in self.columns:
                data_type_dict.update({key: types[key]})
            
            #Assign correct types for specified columns
            self.df.astype(data_type_dict)
            
            #data cleaning (outlier removal + removal of empty columns)
            for key in self.columns:
                if key in optional:
                    masks.append(no_answer_mask(self.df[key]))
                elif key in numeric_safe:
                    masks.append(clean(self.df[key])) 
            
            if len(masks) > 0:
                full_mask = make_mask(tuple(masks))
                self.df = self.df.loc[full_mask, :]

            #print("len dataframe after masking: {0}".format(self.__len__()))
            #calculate pmv index
            #self.data_frame["pmv_index"] = pmv(self.data_frame["pmv_index"], self.data_frame["t_a"], self.data_frame["rh_a"])
            
            #normalize where necessary
            if self.augment_data:
                print("Augmenting data..")
                for key in self.columns:
                    if key in numeric_safe:
                        self.df[key] = self.df[key] + noise(self.df[key].shape)

                if self.continuous_labels:
                    print("Using continuous labels.")
                    self.df["Label"] = self.df["Label"] + noise(self.df["Label"], masked=True)
                
            
        
        if isinstance(self.preprocessing_config, bool) and self.preprocessing_config:
            for key in self.columns:
                func = operations[key]
                p = params[key]
                args = [self.df[key]]
                args.extend(p)
                #print(args)
                self.df[key] = func(*args)
        
        if self.use_imgs:
            paths = list(self.df["RGB_Frontal_View"])
            self.df.pop("RGB_Frontal_View")
            self.img_list = [cv2.resize(cv2.imread(path), self.out_size, interpolation="INTER_NEAREST") for path in tqdm(paths)]
        
        print("Pre-processing done!\r\n")
    
    def train_transform(self, rgb):
        if not rgb is None:
            angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
            do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

            rgb = rotate(angle, rgb)
            rgb = center_crop(rgb, self.output_size)
            rgb = horizontal_flip(rgb, do_flip)
            rgb = np.asfarray(rgb, dtype='float') / 255
        
        return rgb
    
    def val_transform(self, rgb):
        #normed, keypoints, rgb = features

        rgb = center_crop(rgb, self.output_size)
        rgb = np.asfarray(rgb, dtype='float') / 255

        
        return rgb
    
    def __getitem__(self, index):
        """
        Creates input,label pair with data found in each data frame at
        the given index.

        Args:
            index (int): the index to get the data from

        Returns:
            sequence or single input variables, single label
        """
 
        if self.use_imgs: return self.__img_return__(index)
        else: return self.__csv_only_return__(index)
            
    def __img_return__(self, index):
        limit = index+1
        # print(len(self.df["Label"]))
        # print(self.__len__())
        label = np.array(self.df.iloc[[index], -1])
        if self.use_sequence:
            #print("dataframe len: {0}".format(len(self.data_frame["age"])))
            if index+self.sequence_size < self.__len__():
                window = index+self.sequence_size
                limit = window
                label = np.array(self.df.iloc[limit, -1])
            else: 
                limit = self.__len__()-1
                label = np.array(self.df.iloc[limit, -1])
                if index == limit:
                    limit += 1
        
        rgb_input = None
        if self.use_imgs:
            rgb = self.img_list[index]
            if self.use_sequence: rgb = self.img_list[index:limit]
            
            if self.transform is not None:
                rgb_np = self.transform(rgb)
            else:
                raise (RuntimeError("transform not defined"))
            rgb_input = to_tensor(rgb_np)
        
        out = None
        out = torch.from_numpy(np.array(self.df.iloc[[index], :-1])), rgb_input
        label = torch.from_numpy(label)
        # print(out)
        # print(label)
        if self.use_sequence:
            out = torch.from_numpy(np.array(self.df.iloc[index:limit, :-1]))
            #out = torch.unsqueeze(out, dim=1)
            #out = torch.cat(out, dim=1)
            label = label2idx(label)
            label = torch.from_numpy(label)
            #handles padding in case sequence from file end is taken
            if out.shape[0] < self.sequence_size:
                #print("Size before padding: {0}".format(out.shape))
                pad_range = self.sequence_size-out.shape[0]
                last_sequence_line = torch.unsqueeze(out[-1], dim=0)
                for i in range(0,pad_range):
                    out = torch.cat((out,last_sequence_line), dim=0) 
                #print("Size after padding: {0}".format(out.shape))
            
            
        # print(out)
        # print(label)
        # print(label.shape)
        return out.float(), label.long()#.type(torch.LongTensor)
    
    def __csv_only_return__(self, index):
        limit = index+1
       
        label = np.array(self.df.iloc[[index], -1])
        if self.use_sequence:
            if index+self.sequence_size < self.__len__():
                window = index+self.sequence_size
                limit = window
                label = np.array(self.df.iloc[limit, -1])
            else: 
                limit = self.__len__()-1
                label = np.array(self.df.iloc[limit, -1])
                if index == limit:
                    limit += 1
                    
        out = torch.from_numpy(np.array(self.df.iloc[[index], :-1]))
        label = torch.from_numpy(label)
        
        if self.use_sequence:
            out = torch.from_numpy(np.array(self.df.iloc[index:limit, :-1]))
           
            label = label2idx(label)
            label = torch.from_numpy(label)
            #handles padding in case sequence from file end is taken
            if out.shape[0] < self.sequence_size:
                pad_range = self.sequence_size-out.shape[0]
                last_sequence_line = torch.unsqueeze(out[-1], dim=0)
                for i in range(0,pad_range):
                    out = torch.cat((out,last_sequence_line), dim=0) 
        
        return out.float(), label.long()#.type(torch.LongTensor)
    
    
    def __len__(self):
        """
        Returns the number of total data lines. Columns
        are not returned as they are the same for both, training
        and validation dataset.

        Returns:
            int: number of rows int the dataset
        """        
        #print(self.df.shape[0])
        return self.df.shape[0]
    
    

    
