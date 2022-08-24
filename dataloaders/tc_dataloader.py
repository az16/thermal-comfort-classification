from sklearn.feature_extraction import img_to_graph
from dataloaders.dataset import *
from pathlib import Path
from tqdm import tqdm
from dataloaders.utils import *
from PIL import Image
import pandas as pd 

import torch
import numpy as np

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
    def __init__(self, root, split, scale=7, downsample=None, preprocess=None, sequence_size=10, data_augmentation=False, cols=None):
        self.split = split 
        self.root = Path(root) 
        self.scale = scale
        self.preprocessing_config = preprocess #bool or dict of bools that define which signals to preprocess
        self.augment_data = data_augmentation
        self.sequence_size = sequence_size 
        self.columns = parse_features(cols)
        self.metabolic = 1.0 #in met
        self.air_vel = 0.1 #in m/s
        self.downsample = downsample
        #D:/tcs_study/frontal_participant_10_2022-04-28_15-35-18/
        assert not cols is None, "Specify which columns to use as inputs for training."
        print("Using these features: {0}".format(self.columns))
        
        self.load_file_contents(split)
        
        self.preprocess()
        self.pre_compute_labels() 
        print("Using {} datapoint for split {}".format(len(self.data), self.split))   
    
    def load_file_contents(self, split):
        """
        Loads .csv data as np.array and splits input signals/labels.
        Args:
            root (str): path of the root directory
            split (str): training or validation split string
        """
        
        #find files
        print("Searching for {0} files..".format(self.split))
        file_names =  [] #os.listdir(self.root)
        with open("./dataloaders/splits/{0}_{1}.txt".format(self.split, 60)) as file:
            lines = file.readlines()
            file_names = [line.rstrip() for line in lines]
        assert len(file_names) > 0; "No files found at {0}".format(self.root)
            
        file_names = [self.root/x for x in file_names]
        print("Found {0} {1} files at {2}".format(len(file_names),self.split,self.root))
        
        print("Loading contents..")
        
        self.df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(file_names)])#;print(self.df.shape)
        self.df = narrow_labels(self.df, self.scale)
        if not self.downsample is None:
            self.df = self.df[self.df.index % self.downsample == 0] 
        
        print("File contents loaded!")
        
    def preprocess(self):
        """
        Function that assigns appropriate type for each input modality and
        performs preprocessing steps like data cleaning, one-hot-encoding
        and normalization to [0,1]
        """
        print("Pre-processing..")
        #print(self.df.shape)
        masks = []
        if not self.columns == image_only:
            #Load types
            print("Converting .csv strings into correct dtypes..")
            data_type_dict = dict({})
            for key in self.columns:
                data_type_dict.update({key: types[key]})
            
            #Assign correct types for specified columns
            self.df.astype(data_type_dict)
            
            #print(self.df.shape)
            if isinstance(self.preprocessing_config, bool) and self.preprocessing_config:
        
                print("Outlier removal..")
                
                #data cleaning (outlier removal + removal of empty columns)
                
                for key in self.columns:
                    if key in optional:
                        masks.append(no_answer_mask(self.df[key]))
                    elif key in high_outliers:
                        masks.append(clean(self.df[key])) 
                
                if len(masks) > 0:
                    full_mask = make_mask(tuple(masks))
                    self.df = self.df.loc[full_mask, :]

                for key in grouped_removal:
                    if key in self.columns:  
                        self.df = remove_grouped_outliers(group='Label', col=key, df=self.df)
            print(np.sum((self.df["Label"]==-1)))
            print(np.sum((self.df["Label"]==0)))
            print(np.sum((self.df["Label"]==1)))
            #print("len dataframe after masking: {0}".format(self.__len__()))
            #calculate pmv index
            #assert check_pmv_vars(self.columns); "Can't calculate pmv index as not all values are included in the dataframe"
            #if self.use_pmv:
            #    self.df["pmv_index"] = pmv(self.df["Radiation-Temp"], self.df["Clothing-Level"], self.df["PCE-Ambient-Temp"], self.df["Ambient_Humidity"])
            
            #normalize where necessary
        if self.augment_data:
            print("Augmenting data..")
            for key in self.columns:
                if key in numeric_safe:
                    #if not (self.use_col_as_label and self.col_label == key):
                    #print(noise(self.df[key].shape))
                    self.df[key] = self.df[key] + noise(self.df[key].shape)
                
        
        if isinstance(self.preprocessing_config, bool) and self.preprocessing_config:
            print("Normalizing..")
            for key in self.columns:
                #if not (self.use_col_as_label and self.col_label == key):
                func = operations[key]
                p = params[key]
                args = [self.df[key]]
                args.extend(p)
                #print(args)
                self.df[key] = func(*args)
        
        print("Pre-processing done!\r\n")

    
    def __getitem__(self, index):
        """
        Creates input,label pair with data found in each data frame at
        the given index.
        Args:
            index (int): the index to get the data from
        Returns:
            sequence or single input variables, single label
        """

        return self.__csv_only_return__(index)

    def pre_compute_labels(self):
        class_indices = []
        for index in tqdm(range(self.df.shape[0]-self.sequence_size), desc="Precompute labels..."):
            start = index
            stop = start + self.sequence_size
            y = np.asarray(self.df.iloc[start:stop, -1], dtype=int)
            if not len(np.unique(y)) == 1: continue
            class_indices.append([index, y[0]])

        class_indices = np.array(class_indices)
        print("Found {} data points".format(len(class_indices)))
        class_ids, counts = np.unique(class_indices[:,1], return_counts=True)
        print("Class ids: ", class_ids)

        
        for i, c in enumerate(counts):
            print("Class {} counts: {} ({}%)".format(i, c, np.round(c / class_indices.shape[0], 3) * 100))

        class_min = np.argmin(counts)
        count_min = counts[class_min]
        print("Equalizing classes to {} with count {}".format(class_min, count_min))

        new_indices = []
        for cat_idx, cnt in enumerate(counts):
            indices = np.array([[idx, cat_] for [idx, cat_] in class_indices if cat_ == class_ids[cat_idx]])
            stride = np.round(cnt / count_min)
            stride = int(stride)
            indices = indices[::stride]
            new_indices += indices.tolist()
        new_indices = np.array(new_indices)
        class_ids_new, counts = np.unique(new_indices[:,1], return_counts=True)
        for i, c in enumerate(counts):
            print("Class {} counts: {} ({}%)".format(i, c, np.round(c / new_indices.shape[0], 3) * 100))
        assert np.all(class_ids == class_ids_new), "equalizing srewed class ids!!"
        self.data = new_indices
    
    def __csv_only_return__(self, index):
        """
        This function defines how input and labels are returned if only csv values
        are supposed to be used for training.


        Args:
            index: the index in the dataframe to draw data from
        """

        n_feature = np.array(self.df.iloc[0, :-1]).shape[0]
        sequence_size = self.sequence_size
        if self.augment_data and np.random.rand() > 0.5:
            sequence_size = np.random.randint(30, self.sequence_size+1)
        X = torch.zeros((sequence_size, n_feature))

        [start, Y] = self.data[index]
        stop = start + sequence_size
        
        x = np.asarray(self.df.iloc[start:stop, :-1], dtype=np.float32)
        x = torch.from_numpy(x)
        N = len(x)
        X[0:N] = x

        Y_class = label2idx(Y, scale=self.scale)
        Y_order = order_representation(Y_class, scale=self.scale)

        return X, Y_class, Y_order
    
    def __len__(self):
        """
        Returns the number of total data lines. Columns
        are not returned as they are the same for both, training
        and validation dataset.
        Returns:
            int: number of rows int the dataset
        """        
        return len(self.data)
    
    

if __name__ == '__main__':
    dataset = TC_Dataloader(root="F:/data/ThermalDataset", split="training", preprocess=True, data_augmentation=False, sequence_size=60, cols=Feature.BEST, downsample=10, scale=7)
    for (x, y_class, y_order) in dataset:
        print(x, y_class, y_order)
        break
        