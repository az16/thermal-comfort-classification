from sklearn import preprocessing
from dataloaders.dataset import *
from dataloaders.path import Path
from tqdm import tqdm
from dataloaders.utils import *
from dataloaders.path import Path
import pandas as pd 

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
# cols=["Age","Gender","Weight","Height","Bodytemp","Sport-Last-Hour","Time-Since-Meal","Tiredness","Clothing-Level","Radiation-Temp",
#                                                                                                         "PCE-Ambient-Temp",	
#                                                                                                         "Wrist_Skin_Temperature",
#                                                                                                         "Heart_Rate",
#                                                                                                         "GSR",
#                                                                                                         "Ambient_Humidity",
#                                                                                                         "Label"]):

class TC_Dataloader():
    """
    Loads .csv data and preprocesses respective splits
    """
    def __init__(self, split_size=0.8, by_file = True, full=False, cols=["Age","Gender","Weight","Height","Bodytemp","Bodyfat","Sport-Last-Hour","Time-Since-Meal","Tiredness","Clothing-Level","Radiation-Temp",
                                                                                                        "PCE-Ambient-Temp",	
                                                                                                        "Wrist_Skin_Temperature",
                                                                                                        "Heart_Rate",
                                                                                                        "GSR",
                                                                                                        "Ambient_Humidity",
                                                                                                        "Label"]):
        self.columns = cols
        self.independent = cols[:-1]
        self.dependent = cols[-1]
        self.full_set = full
        self.by_file = by_file
        self.split_size = split_size
        assert not cols is None, "Specify which columns to use as inputs for training."
        print("Using these features: {0}".format(self.columns))
        if self.full_set:
            self.load_all()
        elif self.by_file:
            self.load_and_split()
        else:
            self.load_and_split_full()
        
    def load_all(self):
        split = "training"
        print("Searching for {0} files..".format(split))
        train_file_names = os.listdir(Path.db_root_dir("tcs"))
        # with open("./dataloaders/splits/{0}_{1}.txt".format(split, "no_test")) as file:
        #     lines = file.readlines()
        #     train_file_names = [line.rstrip() for line in lines]
        assert len(train_file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
            
        train_file_names = [Path.db_root_dir("tcs")+x for x in train_file_names]
        print("Found {0} {1} files at {2}".format(len(train_file_names),split,Path.db_root_dir("tcs")))
        
        #load .csv contents as list
        print("Loading contents..")
        self.train_df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(train_file_names)])
        print("Creating data frames..")
        
        data_type_dict = dict({})
        #print(types_sk.keys())
        for key in self.columns:
            #print(key)
            data_type_dict.update({key: types_sk[key]})
        
        self.train_df.astype(data_type_dict)
        #self.test_df.astype(data_type_dict)
        
        #shuffle
        #self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)
        # self.test_df = pd.get_dummies(self.test_df)
        
        self.preprocess()
        
        #print(self.train_df)
        self.all_X = self.train_df[self.independent]
        self.all_Y = self.train_df[self.dependent]
        
    def load_and_split(self):
        """
        Loads .csv data as np.array and splits input signals/labels.


        Args:
            root (str): path of the root directory
            split (str): training or validation split string
        """
        
        #find files
        split = "training"
        print("Searching for {0} files..".format(split))
        train_file_names = [] #os.listdir(Path.db_root_dir("tcs"))
        with open("./dataloaders/splits/{0}_{1}.txt".format(split, "no_test")) as file:
            lines = file.readlines()
            train_file_names = [line.rstrip() for line in lines]
        assert len(train_file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
            
        train_file_names = [Path.db_root_dir("tcs")+x for x in train_file_names]
        print("Found {0} {1} files at {2}".format(len(train_file_names),split,Path.db_root_dir("tcs")))
        
        split = "validation"
        print("Searching for {0} files..".format(split))
        test_file_names = [] #os.listdir(Path.db_root_dir("tcs"))
        with open("./dataloaders/splits/{0}_{1}.txt".format(split, "no_test")) as file:
            lines = file.readlines()
            test_file_names = [line.rstrip() for line in lines]
        assert len(test_file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
            
        test_file_names = [Path.db_root_dir("tcs")+x for x in test_file_names]
        print("Found {0} {1} files at {2}".format(len(test_file_names),split,Path.db_root_dir("tcs")))
        
        n = 2
        skip_func = lambda x: x%n != 0
        #load .csv contents as list
        print("Loading contents..")
        self.train_df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(train_file_names)])
        self.test_df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(test_file_names)])
        print("Creating data frames..")
        print(len(self.train_df))
        data_type_dict = dict({})
        #print(types_sk.keys())
        for key in self.columns:
            #print(key)
            data_type_dict.update({key: types_sk[key]})
        
        self.train_df.astype(data_type_dict)
        self.test_df.astype(data_type_dict)
        
        #shuffle
        # self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)
        # self.test_df = self.train_df.sample(frac=1).reset_index(drop=True)
        if "Gender" in self.columns or "Bodyfat" in self.columns:
            self.preprocess()
        
        #print(self.train_df)
        self.train_X = self.train_df[self.independent]
        self.test_X = self.test_df[self.independent]
        self.train_Y = self.train_df[self.dependent]
        self.test_Y = self.test_df[self.dependent]
        
        print("Done.\r\n")
    
    def load_and_split_full(self):
        split = "training"
        print("Searching for {0} files..".format(split))
        train_file_names = os.listdir(Path.db_root_dir("tcs"))
        # with open("./dataloaders/splits/{0}_{1}.txt".format(split, "no_test")) as file:
        #     lines = file.readlines()
        #     train_file_names = [line.rstrip() for line in lines]
        assert len(train_file_names) > 0; "No files found at {0}".format(Path.db_root_dir("tcs"))
            
        train_file_names = [Path.db_root_dir("tcs")+x for x in train_file_names]
        print("Found {0} {1} files at {2}".format(len(train_file_names),split,Path.db_root_dir("tcs")))
        
        #load .csv contents as list
        print("Loading contents..")
        self.full_df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(train_file_names)])
        self.len = len(self.full_df)
        print("Creating data frames..")
        
        data_type_dict = dict({})
        #print(types_sk.keys())
        for key in self.columns:
            #print(key)
            data_type_dict.update({key: types_sk[key]})
        
        self.full_df.astype(data_type_dict)
        limit = int(self.len * self.split_size)
        self.train_df = self.full_df[0:limit]
        self.test_df = self.full_df[limit:]
        # print(self.train_df)
        print(self.test_df)
        #self.test_df.astype(data_type_dict)
        
        #shuffle
        #self.train_df = self.train_df.sample(frac=1).reset_index(drop=True)
        # self.test_df = pd.get_dummies(self.test_df)
        if "Gender" in self.columns or "Bodyfat" in self.columns:
            self.preprocess()
        
        #print(self.train_df)
        self.train_X = self.train_df[self.independent]
        self.test_X = self.test_df[self.independent]
        self.train_Y = self.train_df[self.dependent]
        self.test_Y = self.test_df[self.dependent]
    
    def preprocess(self):
        # self.train_df = emotion2Id(self.train_df).astype({"Emotion-ML": np.int64})
        # self.test_df = emotion2Id(self.test_df).astype({"Emotion-ML": np.int64})
        #print(self.train_df.columns.values.tolist())
        no_answer_train = self.train_df["Bodyfat"] == "No Answer"
        no_answer_train = ~no_answer_train
        self.train_df = self.train_df[no_answer_train]
        self.train_df = convert_str_nominal(self.train_df)
        if not self.full_set:
            no_answer_test = self.test_df["Bodyfat"] == "No Answer"
            no_answer_test = ~no_answer_test
            self.test_df = self.test_df[no_answer_test]
            self.test_df = convert_str_nominal(self.test_df)
        # print(np.array(self.train_df["Tiredness"]).dtype)
        # # self.train_df["Tiredness"] = one_hot(np.array(self.train_df["Tiredness"]), classes=10)
        # # self.test_df["Tiredness"] = one_hot(np.array(self.train_df["Tiredness"]), classes=10)
        # self.train_df["Emotion-ML"] = one_hot(np.array(self.train_df["Emotion-ML"]), classes=7)
        # self.test_df["Emotion-ML"] = one_hot(np.array(self.test_df["Emotion-ML"]), classes=7)
            
    def splits(self):
        if self.full_set:
            return self.all_X, self.all_Y
        return self.train_X, self.train_Y, self.test_X, self.test_Y 
    
    
    

    
