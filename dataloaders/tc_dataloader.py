from sklearn.feature_extraction import img_to_graph
from dataloaders.dataset import *
from dataloaders.path import Path
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
    def __init__(self, root, split, downsample=None, preprocess=None, use_sequence=False, sequence_size=10, crop_size=(1000, 1000),output_size=(224, 224), continuous_labels=False, data_augmentation=False, cols=None, image_path=None, use_imgs=False, label_col=None, forecasting=0):
        self.split = split 
        self.root = root 
        self.forecasting = forecasting
        self.preprocessing_config = preprocess #bool or dict of bools that define which signals to preprocess
        self.augment_data = data_augmentation
        self.output_size = output_size
        self.crop_size = crop_size
        self.use_imgs = use_imgs
        self.img_path = image_path
        self.use_col_as_label = not label_col is None
        self.col_label = None 
        if self.use_col_as_label: self.col_label = label_col
       #self.use_pmv = use_pmv
        self.use_sequence = use_sequence
        self.sequence_size = sequence_size 
        self.columns = [header[x] for x in cols]
        if self.use_imgs and "RGB_Frontal_View" not in self.columns: self.columns.insert(-2, "RGB_Frontal_View")
        if self.use_col_as_label and "Label" in self.columns: self.columns.pop(-1); self.columns.append(self.col_label)
        self.continuous_labels = continuous_labels
        self.metabolic = 1.0 #in met
        self.air_vel = 0.1 #in m/s
        self.downsample = downsample
        #D:/tcs_study/frontal_participant_10_2022-04-28_15-35-18/
        assert not cols is None, "Specify which columns to use as inputs for training."
        print("Using these features: {0}".format(self.columns))
        
        self.rgb_transform = None 
        self.img_list = None 
        if self.use_imgs:
            if self.split == "training": self.rgb_transform = self.train_transform
            else: self.rgb_transform = self.val_transform
            
        self.load_file_contents(split)
        
        
        #do image data-augmentation in case images should be used
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
        file_names =  [] #os.listdir(Path.db_root_dir("tcs"))
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
        if self.use_imgs:
            frames = [pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(file_names)]
            frames = [x.drop(x.tail(1000).index) for x in frames]
            self.df = pd.concat(frames)
        else: self.df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(file_names)])
        if not self.downsample is None:
            self.df = self.df[self.df.index % self.downsample == 0] 
        if self.use_col_as_label:
            self.df.rename({self.col_label:"Label"})
        print("Creating data frames..")
        
        print("File contents loaded!")
        
    def preprocess(self):
        """
        Function that assigns appropriate type for each input modality and
        performs preprocessing steps like data cleaning, one-hot-encoding
        and normalization to [0,1]
        """
        print("Pre-processing..")
        masks = []
        if not self.columns == image_only:
            #Load types
            print("Converting .csv strings into correct dtypes..")
            data_type_dict = dict({})
            for key in self.columns:
                data_type_dict.update({key: types[key]})
            
            #Assign correct types for specified columns
            self.df.astype(data_type_dict)
        
            if isinstance(self.preprocessing_config, bool) and self.preprocessing_config:
        
                print("Outlier removal..")
                
                for key in numeric_safe:
                    if key in self.columns:  
                        self.df = remove_grouped_outliers(group='Label', col=key, df=self.df)
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

        if self.continuous_labels:
            print("Using continuous labels.")
            self.df["Label"] = self.df["Label"] + noise(self.df["Label"], masked=True)
                
            
        
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
        
        if self.use_imgs:
            if self.split == 'validation': 
                self.df["RGB_Frontal_View"]=self.df["RGB_Frontal_View"].replace({'_6_': '_5_'}, regex=True)
            paths = list(self.df["RGB_Frontal_View"].replace({'./images/study/': self.img_path}, regex=True))
            #print(paths)
                
            # print(paths[0])
            # print(os.path.exists(paths[0]))
            self.df.pop("RGB_Frontal_View")
            self.img_list = paths
        
        # for col in self.columns:
        #     print(self.df[col])
        #     print(self.df[col].values.dtype)
            
        print("Pre-processing done!\r\n")
    
    def train_transform(self, rgb):
        if not rgb is None:
            angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
            do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

            rgb = center_crop(rgb, self.crop_size)
            rgb = cv2.resize(rgb, self.output_size, interpolation=cv2.INTER_AREA)
            rgb = rotate(angle, rgb)
            rgb = horizontal_flip(rgb, do_flip)
            rgb = np.asfarray(rgb, dtype='float') / 255
        
        return rgb
    
    def val_transform(self, rgb):

        rgb = center_crop(rgb, self.crop_size)
        rgb = cv2.resize(rgb, self.output_size, interpolation=cv2.INTER_AREA)
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
            rgb = np.asarray(Image.open(rgb).convert('RGB'), dtype=np.uint8)
            if self.use_sequence: 
                rgb = self.img_list[index:limit]
                rgb = [np.asarray(Image.open(x).convert('RGB'), dtype=np.uint8) for x in rgb]
                if len(rgb) < self.sequence_size:
                    #print("Size before padding: {0}".format(out.shape))
                    pad_range = self.sequence_size-len(rgb)
                    last_rgb = rgb[-1]
                    pad = [last_rgb for x in range(0,pad_range)]
                    rgb.extend(pad)
            
            if not self.rgb_transform is None and not self.use_sequence:
                rgb_np = self.rgb_transform(rgb)
                rgb_input = to_tensor(rgb_np)
            elif not self.rgb_transform is None and self.use_sequence:
                rgb_input = [torch.unsqueeze(to_tensor(self.rgb_transform(x)), dim=0) for x in rgb]
                #print(rgb_input[0].shape)
                rgb_input = torch.cat(rgb_input, dim=0)
                #rgb_input = to_tensor(rgb_np)
            else:
                raise (RuntimeError("transform not defined"))
        
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
            label = order_representation(label)
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
        return (rgb_input, out.float()), label.long()#.type(torch.LongTensor)
    
    def __csv_only_return__(self, index):
        limit = index+1
       
        label = np.array(self.df.iloc[[index], -1])
        if self.use_sequence:
            if index+self.sequence_size+self.forecasting < self.__len__():
                window = index+self.sequence_size+self.forecasting
                limit = window
                label = np.array(self.df.iloc[limit, -1])
            else: 
                limit = self.__len__()-1
                label = np.array(self.df.iloc[limit, -1])
                if index == limit:
                    limit += 1
        #print(np.array(self.df.iloc[[index], :-1]).dtype)
        out = torch.from_numpy(np.array(self.df.iloc[[index], :-1]))
        label = torch.from_numpy(label)
        
        if self.use_sequence:
            out = torch.from_numpy(np.array(self.df.iloc[index:limit, :-1]))

            if not self.use_col_as_label:
                label = label2idx(label)
                label = order_representation(label)
                label = torch.from_numpy(label)
            #handles padding in case sequence from file end is taken
            if out.shape[0] < self.sequence_size:
                pad_range = self.sequence_size-out.shape[0]
                last_sequence_line = torch.unsqueeze(out[-1], dim=0)
                for i in range(0,pad_range):
                    out = torch.cat((out,last_sequence_line), dim=0) 
         
        return out, label#.type(torch.LongTensor)
    
    
    def __len__(self):
        """
        Returns the number of total data lines. Columns
        are not returned as they are the same for both, training
        and validation dataset.
        Returns:
            int: number of rows int the dataset
        """        
        #print(self.df.shape[0])
        return self.df.shape[0]-(self.sequence_size+self.forecasting)
    
    

