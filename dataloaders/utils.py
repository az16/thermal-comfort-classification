
import pickle
from PIL import Image
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic, met_typical_tasks
import scipy.ndimage.interpolation as itpl
from itertools import permutations as p 
import cv2 
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os 


types = dict({
            "Age" : np.float32,
            "Gender" : str,
            "Weight": np.float32,
            "Height": np.float32,
            "Bodyfat": str,
            "Bodytemp": np.float32,
            "Sport-Last-Hour": np.int64,
            "Time-Since-Meal": str,
            "Tiredness": np.int64,
            "Clothing-Level": np.float32,
            "Radiation-Temp": np.float32,
            "PCE-Ambient-Temp":np.float32,
            "RGB_Frontal_View" : str,
            "Nose": str,
            "Neck": str,
            "RShoulder": str,
            "RElbow": str,
            "LShoulder": str,
            "LElbow": str,
            "REye": str,
            "LEye": str,
            "REar": str,
            "LEar" : str,
            "Heart_Rate" : np.float32,
            "Wrist_Skin_Temperature" : np.float32,
            "GSR" : np.float32,
            "Ambient_Temperature" : np.float32,
            "Ambient_Humidity" : np.float32,
            "Emotion-ML" : str,
            "Emotion-Self" : np.float32,
            "Label" : np.float32})

types_sk = dict({
            "Age" : np.float32,
            "Gender" : 'category',
            "Weight": np.float32,
            "Height": np.float32,
            "Bodyfat": str,
            "Bodytemp": np.float32,
            "Sport-Last-Hour": bool,
            "Time-Since-Meal": np.int32,
            "Tiredness": np.int64,
            "Clothing-Level": np.float32,
            "Radiation-Temp": np.float32,
            "PCE-Ambient-Temp":np.float32,
            "Radiation-Temp":np.float32,
            "Heart_Rate" : np.float32,
            "Wrist_Skin_Temperature" : np.float32,
            "GSR" : np.float64,
            "Ambient_Temperature" : np.float32,
            "Ambient_Humidity" : np.float32,
            "Emotion-ML" : str,
            "Label" : np.float32,
            "RGB_Frontal_View": np.float64})

header = ["Timestamp",
          "Age",
          "Gender",
          "Weight",
          "Height",
          "Bodyfat",
          "Bodytemp",
          "Sport-Last-Hour",
          "Time-Since-Meal",
          "Tiredness",
          "Clothing-Level",
          "Radiation-Temp",
          "PCE-Ambient-Temp",
          "Air-Velocity",
          "Metabolic-Rate",
          "Emotion-Self",
          "Emotion-ML",	
          "RGB_Frontal_View",
          "Nose",
          "Neck",
          "RShoulder",
          "RElbow",
          "LShoulder",
          "LElbow",
          "REye",
          "LEye",
          "REar",
          "LEar",
          "Wrist_Skin_Temperature",
          "Heart_Rate",
          "GSR",
          "Ambient_Temperature",
          "Ambient_Humidity",
          "Label"]

numeric_safe = [ "Ambient_Temperature","Radiation-Temp", "Ambient_Humidity", "Wrist_Skin_Temperature", "Heart_Rate", "GSR"]#, "Ambient_Humidity","Heart_Rate", "Wrist_Skin_Temperature", "GSR"]
high_outliers = ["Heart_Rate","GSR"]
grouped_removal = ["Ambient_Temperature","Radiation-Temp",]
numeric_unsafe = ["Age", "Time-Since-Meal", "Bodytemp", "Weight", "Height", "Bodyfat"]
categorical = ["Tiredness", "Emotion-ML", "Emotion-Self"]
binary = ["Sport-Last-Hour"]
optional = ["Weight", "Height", "Bodyfat"]
image_only = ["RGB_Frontal_View"]

def narrow_labels(df, scale=2):
        """
        Reduces the 7 point ts scale to either 3 or 2 labels.


        Args:
            df: the label column of the dataframe
            scale: the scale to reduce to
        """
        #df = df[~df.Label==0]
        #print(df["Label"])
        if scale==2:
            print("2-Point scale transformation.")
            # x = df["Label"] == 0
            # x = ~x
            # df = df[x]
            # x = df["Label"] == 1
            # x = ~x
            # df = df[x]
            df.loc[(df["Label"] == 0), "Label"] = 1
            df.loc[(df["Label"] == -3), "Label"] = 0
            df.loc[(df["Label"] == -2), "Label"] = 0
            df.loc[(df["Label"] == -1), "Label"] = 1
            df.loc[(df["Label"] == 1), "Label"] = 1
            df.loc[(df["Label"] == 2), "Label"] = 0
            df.loc[(df["Label"] == 3), "Label"] = 0
            #print(df)
            #ck = df.groupby("Label")
            return df
        elif scale==3:
            print("3-Point scale transformation.")
           
            # rescale = {-1: 0, -3:-1, -2:-1, 1:0, 2:1, 3:1}
            # df["Label"].replace(rescale, inplace=True)
            df["Label"] = df["Label"] + noise(df["Label"], std=0.5, masked=True)
            #print(np.sum((df["Label"]==-3))+np.sum((df["Label"]==-2)))
            df.loc[((df["Label"] >= -0.5) & (df["Label"] <= 0.5)), "Label"] = 0
            df.loc[(df["Label"] > 0.5), "Label"] = 1
            df.loc[(df["Label"] < -0.5), "Label"] = -1
            # print(np.sum((df["Label"]==-1))+np.sum((df["Label"]==1))+np.sum((df["Label"]==0)))
            # print(np.sum((df["Label"]==2))+np.sum((df["Label"]==3)))
            # #df.loc[(df["Label"] == -1), "Label"] = 0
            # df.loc[(df["Label"] == -3), "Label"] = -1
            # df.loc[(df["Label"] == -2), "Label"] = -1
            # df.loc[(df["Label"] == 0), "Label"] = 0
            # #df.loc[(df["Label"] == 1), "Label"] = 0
            # df.loc[(df["Label"] == 2), "Label"] = 1
            # df.loc[(df["Label"] == 3), "Label"] = 1
            print(np.sum((df["Label"]==-1)))
            print(np.sum((df["Label"]==0)))
            print(np.sum((df["Label"]==1)))
            #print(df)
            return df
        return df
            

    
def rgb_loader(paths):
    """
        Loads images given a list of paths.


        Args:
            paths: list of rgb paths
    """
    #assert os.path.exists(file), "file not found: {}".format(root + file)
    return [np.asarray(Image.open(file).convert('RGB'), dtype=np.uint8) for file in paths]

def one_hot(sample, classes=3):
    """
        Takes a categorical column and turns it into a one hot vector column.


        Args:
            sample: the categorical column
            classes: the number of classes for oneh ot encoding
    """
    r = np.zeros((sample.size, classes))
    r[np.arange(sample.shape[0]), sample] = 1
    return r

def to_keypoint(sample):
    """
        Takes a key point column and turns it into a numpy vector of length 3 (x,y,z).


        Args:
            sample: the key point column
    """
    sample = np.char.split(sample, sep="~")
    # r = np.array([x.split('~') for x in sample]).astype(np.float32)     
    # return r
    return sample

def to_tensor(img):
    """
        Takes a pil image and turns it into a pytorch tensor.


        Args:
            img: the input pil image to transform.
    """
    # if img.ndim == 4:
    #     img = torch.from_numpy(img.transpose((0, 3, 1, 2)).copy()) #return sequence of 3-channel torch tensors
    if img.ndim == 3:
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()) #return single 3-channel torch tensor
    elif img.ndim == 2:
        img = torch.from_numpy(img.copy())
    
    return img.float()

def norm(sample, min=None, max=None):
    """
        Does min-max normalization with customizable min and max limits.


        Args:
            sample: the data column
            min: customizable minimum value in the column
            max: customizable maximum value in the column
    """
    if not min is None and max is None:
        max = np.max(sample)
    elif not max is None and min is None:
        min = np.min(sample)
    elif max is None and min is None:
        min = np.min(sample)
        max = np.max(sample)
    return (sample - min)/(max-min) 
    

def standardize(sample):
    """
        Standardizes a data column using mean and std computations.


        Args:
            sample: the data column
    """
    return (sample-np.mean(sample)/np.std(sample))

def clean(sample, missing_id=0, cutoff_multiplier = 1.5):
    """
        Outlier removal using the mean and std. Any value outside
        3 stds is considered an outlier by this method


        Args:
            sample: the data column
    """
    # q25, q75 = np.percentile(sample,[25, 75])
    # iqr = q75-q25 
    # cutoff = iqr * cutoff_multiplier
    #lower, upper = q25-cutoff, q75+cutoff
    lower, upper = np.mean(sample)-3*np.std(sample),np.mean(sample)+3*np.std(sample)
    mask_u = (sample <= upper)
    mask_l = (sample >= lower) 
    #mask_z = (sample > 0) 
    mask = np.logical_and(mask_u, mask_l)
    return mask

def no_answer_mask(frame, col=False):
    """
        Deletes "No Answer" values from the dataframe.


        Args:
            frame: the data frame
            col: the column to delete values from
    """
    if not col:
        return (frame != "No Answer")

#This method uses the PyThermal package available at: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort
def pmv(radiation, clothing, t_a, rh_a):
    """
        Extracts necessary values for PMV computation and computes the PMV index.


        Args:
            radiation: the radiation temperature column
            clothing: the clothing column
            t_a: the ambient temperature column
            rh_a: the relative humidity column
    """
    #print(rh_a)
    # col_length = len(t_a)
    tdb = t_a.values 
    tr = radiation.values
    icl = clothing.values
    v = np.ones(t_a.shape)*0.1
    rh = rh_a.values
    met = np.ones(t_a.shape)
    
    vr = v_relative(v, met)
    #clo = clo_dynamic(icl, met)
    #print(tdb[0], tr[0], vr[0], rh[0], met[0], icl[0])
    
    pmv = pmv_ppd(tdb, tr, vr, rh, met, icl, standard="ASHRAE")["pmv"]
    #print(np.isnan(pmv).any())
    #print(pmv_ppd(tdb[0], tr[0], vr[0], rh[0], met[0], icl[0], standard="ASHRAE")["pmv"])
    #print(pmv)
    #return np.array([pmv_ppd(tdb[x], tr[x], vr[x], rh[x], met[x], clo[x])["pmv"] for x in range(0, col_length)])
    return pmv

def rotate(angle, imgs, reshape=False, prefilter=False, order=0):
    """
        Rotates and image given a rotation angle.


        Args:
            angle: how much to rotate in degrees
            img: the image to rotate
            reshape: should the image be reshaped?
            prefilter: should a filter be applied before rotating?
            order: which axis to start rotating from
    """
    imgs = itpl.rotate(imgs, angle, reshape=reshape, prefilter=prefilter, order=order)
    return imgs

def center_crop(img, output_size):
    """
        Extracts a central crop of an image given the desired crop size.


        Args:
            img: the img to crop
            output_size: the desired crop size
    """
    h = img.shape[1]
    w = img.shape[2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i+h, j:j+w, :]

def horizontal_flip(img, do_flip):
    """
        Flips an img horizontally if flip is true.


        Args:
            img: the img tensor to flip
            flip: bool that denotes whether to flip or not
    """
    if do_flip: return np.fliplr(img[:,])
    else: return img 

def augmentation(x, val=False):
    """
        Takes an img tensor and adds noise from a uniform distribution to the pixels.


        Args:
            x: the img tensor
            val: is it a validation image?
    """
    n, k = x 
    do_scale = np.random.uniform(0.0, 1.1) < 0.5
    s = None 
    if val:
        if do_scale: s = -0.1
        else: s = 0.1
    else:
        if do_scale: s = -1.0
        else: s = 1.0
     
    params_n = np.array([0.5, 0.2, 3.0, 1.0, 1.0, 1.0]) * s
    
    params_k = np.ones_like(k)*(5.0*s) 
    
    if not k is None and n is None:
        return k+params_k
    
    if k is None and not n is None:
         return n+params_n
    
    return (n+params_n, k+params_k)

def make_mask(list_of_masks):
    """
        Concatenates a list of bool tensors using and concatenation.


        Args:
            list_of_masks: the list of masks to be concatenated
    """
    if len(list_of_masks) == 1: return list_of_masks[0]
    init = np.logical_and(list_of_masks[0], list_of_masks[1])
    if len(list_of_masks) > 2:
        for mask in list_of_masks:
            init = np.logical_and(init, mask)
    
    return init 

def convert_str_nominal(x, column=False):
    """
        Turns Male/Female columns into 0/1 columns.


        Args:
            x: either a dataframe or gender column
            column: is it a column or a dataframe?
    """
    if column:
        return (x == "Female").astype(np.int8).astype(np.float64)
    i=0
    for key in ["Male", "Female"]:
        x.loc[(x["Gender"] == key), "Gender"] = i
        i += 1
    return x

def place_holder(x):
    """
        Function added as a skip value for preprocessing. Only returns what it's been given.


        Args:
            x: the data frame column
    """
    return x

def noise(df, mean = 0, std = 0.3, masked=False):
    """
        Adds noise from a gaussian distribution and adds it to a feature column (or labels).


        Args:
            df: the data frame column
            mean: mean parameter for the gaussian noise to sample from
            std: std parameter for the gaussian noise to sample from
            masked: are we adding noise to labels?
    """
    if not masked:
        return np.random.normal(mean, std, df)
    full = np.random.normal(mean, std, df.shape[0])
    tmp = np.array(df)
    tmp = ~((-2 >= tmp)+(tmp >= 2))
    full = full * tmp
    return full

def make_full_path(root, df):
    """
        Constructs the full RGB path from the received csv paths.


        Args:
            root: path to directory where the RGB images are held
            df: the dataframe
    """
    df["RGB_Frontal_View"] = root + df['RGB_Frontal_View'].astype(str)
    return df

def load_resized(out_size, df):
    """
        Loads images as resized images and applies interpolation.


        Args:
            out_size: the size to which images are supposed to be resized to
            df: the dataframe column with all image paths
    """
    df["RGB_Frontal_View"] = pd.Series(np.array([cv2.resize(cv2.imread(path), out_size, interpolation="INTER_NEAREST") for path in df["RGB_Frontal_View"]]))
    return df

operations = dict({
            "Age" : norm,
            "Gender" : convert_str_nominal,
            "Weight": norm,
            "Height": norm,
            "Bodyfat": norm,
            "Bodytemp": norm,
            #"Sport-Last-Hour": convert_binary,
            "Time-Since-Meal": norm,
            "Tiredness": one_hot,
            "Radiation-Temp": norm,
            "PCE-Ambient-Temp": norm,
            "Clothing-Level": place_holder,
            "RGB_Frontal_View": place_holder,
            "Nose": to_keypoint,
            "Neck": to_keypoint,
            "RShoulder": to_keypoint,
            "RElbow": to_keypoint,
            "LShoulder": to_keypoint,
            "LElbow": to_keypoint,
            "REye": to_keypoint,
            "LEye": to_keypoint,
            "REar": to_keypoint,
            "LEar" : to_keypoint,
            "Heart_Rate" : norm,
            "Wrist_Skin_Temperature" : norm,
            "GSR" : norm,
            "Ambient_Temperature" : norm,
            "Ambient_Humidity" : norm,
            "Emotion-ML" : one_hot,
            "Emotion-Self" : one_hot,
            "Label":place_holder})

params = dict({
            "Age" : [0,100],
            "Gender" : [True],
            "Weight": [40.0, 110.0],
            "Height": [140.0, 210.0],
            "Bodyfat": [0.05, 40.0],
            "Bodytemp": [],
            "Sport-Last-Hour": [],
            "Time-Since-Meal": [],
            "Tiredness": [9],
            "Radiation-Temp": [15,35],
            "PCE-Ambient-Temp": [10,40],
            "Clothing-Level": [],
            "Nose": [],
            "Neck": [],
            "RShoulder": [],
            "RElbow": [],
            "LShoulder": [],
            "LElbow": [],
            "REye": [],
            "LEye": [],
            "REar": [],
            "LEar" : [],
            "Heart_Rate" : [40, 130],
            "Wrist_Skin_Temperature" : [],
            "GSR" : [],
            "Ambient_Temperature" : [15,40],
            "Ambient_Humidity" : [0.0, 100.0],
            "Emotion-ML" : [7],
            "Emotion-Self" : [6],
            "Label":[],
            "RGB_Frontal_View":[]})


def emotion2Id(x):
    """
        Maps the emotion labels to numericals (0-6).


        Args:
            x: the emotion column
    """
    i=0
    for key in ["Angry", "Fear", "Disgust", "Neutral", "Surprise", "Happy", "Sad"]:
        x.loc[(x["Emotion-ML"] == key), "Emotion-ML"] = i
        i += 1
    return x
    
def label2idx(label, scale=7):
    """
        Maps labels to their position in the label list depending on scale type.


        Args:
            label: the label column
            scale: the scale type to be used (7,3,2)
    """
    idx = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
    if scale == 2: idx = [0.0,1.0]
    elif scale == 3: idx = [-1.0,0.0,1.0]
    
    return idx.index(label)

def idx2label(idx):
    label = torch.clone(idx)
    label[label == 0] = -3
    label[label == 1] = -2
    label[label == 2] = -1
    label[label == 3] = 0
    label[label == 4] = 1
    label[label == 5] = 2
    label[label == 6] = 3
    return label

def order_representation(label, sklearn=False, scale=7):
    """
        Transforms labels to order representation after Cheng et al. (https://www.researchgate.net/publication/221533108_A_Neural_Network_Approach_to_Ordinal_Regression)


        Args:
            label: the label column
            sklearn: are we using sklearn to train?
            scale: the scale type to be used (7,3,2)
    """
    if sklearn:
        return sk_order_representation(label)
    # print(label)
    # print(label.shape)
    if label == scale-1:
        #print(np.ones(shape=(7)))
        return np.ones(shape=(scale))
    else:
        category_vector = np.ones(shape=(scale))
        category_vector[np.int8(label)+1:] = 0
        #print(category_vector)
        return category_vector
        
def sk_order_representation(labels):
    """
        Transforms labels to order representation after Cheng et al. (https://www.researchgate.net/publication/221533108_A_Neural_Network_Approach_to_Ordinal_Regression).
        Can only be used for sklearn-based training


        Args:
            label: the label column
    """
    np_labels = labels.values
    #np_labels = labels
    #print(labels)
    i = 0
    r = np.ones((np_labels.shape[0], 7))
    for val in np_labels:
        r[i,val:] = 0 
        #print(r[i])
        i += 1
    return r

def remove_grouped_outliers(group, col, df):
    """
        Groups the specified column by label and does outlier removal per group based on mean and std parameters.
        
        Args:
            group: variable to group by
            col: column to do outlier removal on
            df: the dataframe to take the grouped column from
    """
    grouped_df = df.groupby(group)[col]
    labels = [label for label,g_p in grouped_df]
    means = grouped_df.mean().values
    stds = grouped_df.std().values
    lower, upper = means-3*stds, means+3*stds
    bound_index = 0
    for i in labels:
        c_label = i       
        df = df.drop(df[(((df[col] > upper[bound_index]) | (df[col] < lower[bound_index])) & (df[group]==c_label))].index)
        bound_index += 1
    return df

def get_change_rate(df, look_ahead_window=1000):
    """
        Computes gradients for sequential data using a look-ahead window


        Args:
            df: the dataframe to use
            look-ahead-window: how many time steps to consider for computation
    """
    df = df.values
    shifted = np.concatenate((df[look_ahead_window:], np.ones(look_ahead_window)*df[-1]))
    change_rate = shifted-df
    return change_rate

def feature_permutations(features, max_size=5):
    """
        Creates all possible feature permutations based on basic combinatorics


        Args:
            features: the features to find combinations for
            max_size: if only feature combinations up to a certain size are desired use this flag and set it the desired length
    """
    pms = []
    for i in range(2,max_size+1):
        pms.extend(list(p(features, i)))
    pms = list(set([tuple(sorted(x)) for x in pms]))
    return pms
        

def order2class(o):
    o = o.detach()
    # o.shape (B, 7)
    B = o.shape[0]
    N = o.shape[1]
    o = torch.round(o)
    c = torch.zeros(B)
    for b in range(B):
        dim0 = torch.where(o[b]==1.0)[0]
        if len(dim0):
            c[b] = torch.argmax(dim0)
        else:
            c[b] = 0
    c = torch.clip(c, 0, N-1)
    return torch.nn.functional.one_hot(c.long(), N).float().to(o)
        

def class2order(c):    
    c = c.detach()
    if len(c.shape) > 1:
        if torch.sum(c[0]) - 1.0 > 1e-6: 
            c=c.softmax(dim=1)
        # c.shape (B, 7)
        class_idx = torch.argmax(c, dim=1) # (B)
        order_rep = torch.ones_like(c) # (B, 7)
    else:
        class_idx = c
        order_rep = torch.ones((len(c), 7)) # (B, 7)
    for b,idx in enumerate(class_idx):
        order_rep[b, idx+1:] = 0
    return order_rep.to(c)

def class7To3(v):
    numpy = False
    if isinstance(v, np.ndarray):
        numpy = True
        v = torch.from_numpy(v)
    vec = torch.clone(v)
    vec[vec<=2] = 0
    vec[vec==3] = 1
    vec[vec>=4] = 2
    if numpy:
        vec = vec.numpy()
    return vec

def class7To2(v):
    numpy = False
    if isinstance(v, np.ndarray):
        numpy = True
        v = torch.from_numpy(v)
    vec = torch.clone(v)
    vec[vec == 0] = 0
    vec[vec == 1] = 0
    vec[vec == 2] = 1
    vec[vec == 3] = 1
    vec[vec == 4] = 1
    vec[vec == 5] = 0
    vec[vec == 6] = 0
    if numpy:
        vec = vec.numpy()
    return vec