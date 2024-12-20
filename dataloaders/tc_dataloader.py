from dataloaders.dataset import *
from tqdm import tqdm
from dataloaders.utils import *
from PIL import Image
import pandas as pd 

import torch
import numpy as np

# raw data table
# index     name       
#  0        timestamp      
#  1        age                             static                                              
#  2        gender                          static                              
#  3        weight                          static                              
#  4        height                          static                              
#  5        bodyfat                         static                               
#  6        bodytemp                        dynamic                                
#  7        sport                           static                             
#  8        meal                            static                             
#  9        tiredness                       static   
# 10        clothing-level                  static                              
# 11        black-globe-temp                dynamic                                        
# 12        pce-ambient                     dynamic                                   
# 13        air-velocity                    Missing!                                      
# 14        metabolic rate                  static                                      
# 15        emotion                         -                               
# 16        emotion-ml                      -                                  
# 17        rgb paths                       -                                 
# 18-27     key-points                      -                                  
# 28-30     bio-signals                     dynamic                                   
# 31,32     environmental-signals           dynamic                                             
# 33        label


class TC_Dataloader(BaseDataset):
    """
    Loads .csv data and preprocesses respective splits
    Args:
        BaseDataset (Dataset): loads and splits dataset
    """
    def __init__(self, root, split, run=None, scale=7, downsample=None, preprocess=None, use_sequence=False, sequence_size=10, crop_size=(1000, 1000),output_size=(224, 224), continuous_labels=False, data_augmentation=False, cols=None, image_path=None, use_imgs=False, label_col=None, forecasting=0):
        self.split = split 
        self.root = root 
        self.run = run
        self.forecasting = forecasting
        self.scale = scale
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
        if self.split == 'all':
            split_file = "./dataloaders/splits/real_all.txt"
        else:
            split_file = "./dataloaders/splits/{0}_{1}_real.txt".format(self.split, 60)
            if not self.run is None:
                split_file = "./dataloaders/splits/{0}_indices_{1}.txt".format(self.split, self.run)

        file_names =  [] #os.listdir(self.root)
        with open(split_file) as file:
            lines = file.readlines()
            file_names = [line.rstrip() for line in lines]
        assert len(file_names) > 0; "No files found at {0}".format(self.root)
            
        file_names = [self.root+x for x in file_names]
        print("Found {0} {1} files at {2}".format(len(file_names),self.split,self.root))
        
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
        else: self.df = pd.concat([pd.DataFrame(pd.read_csv(x, delimiter=";"), columns = self.columns) for x in tqdm(file_names)])#;print(self.df.shape)
        self.df.dropna(axis=0, inplace=True)
        self.df = narrow_labels(self.df, self.scale)
        if not self.downsample is None:
            # print(self.df.shape)
            # print(self.downsample)
            self.df = self.df[self.df.index % self.downsample == 0] 
            # print(self.df.shape)
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
            self.df.pop("RGB_Frontal_View")
            self.img_list = paths
   
        
        print("Pre-processing done!\r\n")
    
    def train_transform(self, rgb):
        """
        This function defines the data augmentation for training images


        Args:
            rgb: the rgb image tensor to do data augmentation for
        """
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
        """
        This function defines the data augmentation for validation images.

        Args:
            rgb: the rgb image tensor to do data augmentation for
        """
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
        """
        This function defines how input and labels are returned if images are supposed
        to be used for training.


        Args:
            index: the index in the dataframe to draw data from
        """
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
            label = label2idx(label, scale=self.scale)
            label = order_representation(label, scale=self.scale)
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
        return (rgb_input, out.float()), label.float()#.type(torch.LongTensor)
    
    def __csv_only_return__(self, index):
        """
        This function defines how input and labels are returned if only csv values
        are supposed to be used for training.


        Args:
            index: the index in the dataframe to draw data from
        """
        limit = index+1
       
        label = np.array(self.df.iloc[[index], -1])
        if self.use_sequence:
            if index+self.sequence_size+self.forecasting < self.df.shape[0]:
                window = index+self.sequence_size+self.forecasting
                limit = window
                label = np.array(self.df.iloc[limit, -1])
            else: 
                limit = self.self.df.shape[0]-1
                label = np.array(self.df.iloc[limit, -1])
                if index == limit:
                    limit += 1
        #print(np.array(self.df.iloc[[index], :-1]).dtype)
        arr = np.array(self.df.iloc[[index], :-1])
        arr[arr=="Female"] = 0
        arr[arr=="Male"] = 1
        out = torch.from_numpy(arr.astype(float))
        label = torch.from_numpy(label)
        # print(limit)
        # print(limit-self.forecasting < self.__len__())
        
        limit -= self.forecasting
        #         if limit < index:
        #             index -= (self.forecasting+self.sequence_size)
        # else:
        #     limit = self.__len__()-1
        #     if index == limit:
        #         limit += 1
        if self.use_sequence:
            out = torch.from_numpy(np.array(self.df.iloc[index:limit, :-1]))
            # print(out.shape)

            #handles padding in case sequence from file end is taken
            if out.shape[0] < self.sequence_size:
                #print(out.shape, index, (x,limit))
                pad_range = self.sequence_size-out.shape[0]
                last_sequence_line = torch.unsqueeze(out[-1], dim=0)
                for i in range(0,pad_range):
                    out = torch.cat((out,last_sequence_line), dim=0) 

        if self.augment_data and np.random.rand() > 0.5:
            out += torch.randn_like(out) * 0.02

        if not self.use_col_as_label:
            label = label2idx(label, scale = self.scale)
            label = order_representation(label, scale=self.scale)
            label = torch.from_numpy(label)
            
        assert not torch.any(torch.isnan(label)), index
        assert not torch.any(torch.isnan(out)), index
         
        return out.float(), label.float()#.type(torch.LongTensor)
    
    
    def __len__(self):
        """
        Returns the number of total data lines. Columns
        are not returned as they are the same for both, training
        and validation dataset.
        Returns:
            int: number of rows int the dataset
        """        
        # print(self.df.shape[0])
        # print(self.df.shape[0]-(self.sequence_size+self.forecasting))
        return self.df.shape[0]-(self.sequence_size+self.forecasting)
    
    

if __name__ == '__main__':
    #indices = np.arange(19)
    #with open("dataloaders/splits/all.txt", "r") as txtfile:
    #    all_participants = [line.strip() for line in txtfile.readlines()]
    #
    #for i in range(20):
    #    train_indices = sorted(np.random.choice(indices, 16, replace=False))
    #    remaining = [idx for idx in indices if not idx in train_indices]
    #    val_indices = sorted(np.random.choice(remaining, 2, replace=False))
    #    test_indices = [idx for idx in remaining if not idx in val_indices]
    #    with open("dataloaders/splits/training_indices_{}.txt".format(i), "w") as txtfile:
    #        txtfile.writelines([all_participants[idx] + "\n" for idx in train_indices])
    #    with open("dataloaders/splits/validation_indices_{}.txt".format(i), "w") as txtfile:
    #        txtfile.writelines([all_participants[idx] + "\n" for idx in val_indices])
    #    with open("dataloaders/splits/test_indices_{}.txt".format(i), "w") as txtfile:
    #        txtfile.writelines([all_participants[idx] + "\n" for idx in test_indices])
    #dataset = TC_Dataloader("H:/data/ThermalDataset/", "training", cols=[28, 29, 30, 31,32,33])
    dataset = TC_Dataloader("H:/data/ThermalDataset/", "all", cols=[1, 2, 3, 4, 6, 8, 9, 10, 28, 29, 30, 31,32,33], use_sequence=False)
    #dataset = TC_Dataloader("H:/data/ThermalDataset/", "training", run=11, cols=[6,11,12,28,29,30,31,32,33])
    
    N = len(dataset.columns)
    values = [[] for _ in range(N)]
    for (inputs, labels) in tqdm(dataset):
        inputs = inputs[0].cpu().numpy()
        for i in range(len(values)-1):
            if i == 2 and inputs[i] == 0:continue
            values[i].append(inputs[i])
        label = torch.argmax(order2class(labels.unsqueeze(0))).item()
        values[-1].append(label)
            
    for i in range(len(values)):
        wst = values[i]
        name = dataset.columns[i]
        if i == 1:
            wst = np.array(wst).astype(int)
            print(f"Male {np.sum(wst==1)} Female {np.sum(wst==0)}")
        else:
            print(f"{name} {np.mean(wst):0.2f} {np.std(wst):0.2f} {np.min(wst)} {np.max(wst)}")
            
    np.save("all_real.npy", np.array(values, dtype=object))
    data = np.load("all_real.npy", allow_pickle=True)
