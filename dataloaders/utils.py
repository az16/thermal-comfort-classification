
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


types = dict({
            "Age" : np.float32,
            "Gender" : str,
            "Weight": np.float32,
            "Height": np.float32,
            "Bodyfat:": str,
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

numeric_safe = ["PCE-Ambient_Temp", "Radiant-Temp", "Heart_Rate", "Wrist_Skin_Temperature", "GSR", "Ambient_Humidity"]
numeric_unsafe = ["Age", "Time-Since-Meal", "Bodytemp", "Weight", "Height", "Bodyfat"]
categorical = ["Gender", "Tiredness", "Emotion-ML", "Emotion-Self"]
binary = ["Sport-Last-Hour"]
optional = ["Weight", "Height", "Bodyfat"]
image_only = ["RGB_Frontal_View"]


    
def rgb_loader(paths):
    #assert os.path.exists(file), "file not found: {}".format(root + file)
    return [np.asarray(Image.open(file).convert('RGB'), dtype=np.uint8) for file in paths]

def one_hot(sample, classes=3):
    r = np.zeros((sample.size, classes))
    r[np.arange(sample.shape[0]), sample] = 1
    return r

def to_keypoint(sample):
    sample = np.char.split(sample, sep="~")
    # r = np.array([x.split('~') for x in sample]).astype(np.float32)     
    # return r
    return sample

def to_tensor(img):
    # if img.ndim == 4:
    #     img = torch.from_numpy(img.transpose((0, 3, 1, 2)).copy()) #return sequence of 3-channel torch tensors
    if img.ndim == 3:
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy()) #return single 3-channel torch tensor
    elif img.ndim == 2:
        img = torch.from_numpy(img.copy())
    
    return img.float()

def norm(sample, min=None, max=None):
    # print(sample)
    #print(min, max)
    if not min is None and max is None:
        max = np.max(sample)
    elif not max is None and min is None:
        min = np.min(sample)
    elif max is None and min is None:
        min = np.min(sample)
        max = np.max(sample)
    return (sample - min)/(max-min) 
    # x = np.array(sample.values) #returns a numpy array
    # x_scaled = standardize(x)
    # df = pd.DataFrame(x_scaled)
    # return df

def standardize(sample):
    return (sample-np.mean(sample)/np.std(sample))

def clean(sample, missing_id=0, cutoff_multiplier = 1.5):
    q25, q75 = np.percentile(sample,[25, 75])
    iqr = q75-q25 
    cutoff = iqr * cutoff_multiplier
    lower, upper = q25-cutoff, q75+cutoff
    mask_u = (sample <= upper)
    mask_l = (sample >= lower) 
    #mask_z = (sample > 0) 
    mask = np.logical_and(mask_u, mask_l)
    return mask

def no_answer_mask(frame):
    return (frame != "No Answer")

def check_pmv_vars(columns):
    required = ["PCE-Ambient-Temp", "Radiation-Temp", "Clothing-Level", "Ambient_Humidity"]
    for col in required:
        if not col in columns:
            return False 
    
    return True

# TODO: cite pythermal creators
def pmv(radiation, clothing, t_a, rh_a):
    
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
    imgs = itpl.rotate(imgs, angle, reshape=reshape, prefilter=prefilter, order=order)
    return imgs

def center_crop(img, output_size):
    h = img.shape[1]
    w = img.shape[2]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i+h, j:j+w, :]

def horizontal_flip(img, do_flip):
    if do_flip: return np.fliplr(img[:,])
    else: return img 

def augmentation(x, val=False):
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
    
    init = np.logical_and(list_of_masks[0], list_of_masks[1])
    if len(list_of_masks) > 2:
        for mask in list_of_masks:
            init = np.logical_and(init, mask)
    
    return init 

def convert_str_nominal(x):
    i=0
    for key in ["Male", "Female"]:
        x.loc[(x["Gender"] == key), "Gender"] = i
        i += 1
    return x

def convert_binary(x):
    
    return (x == 1)

def place_holder(x):
    return x

def noise(df, mean = 0, std = 0.3, masked=False):
    if not masked:
        return np.random.normal(mean, std, df)
    full = np.random.normal(mean, std, df.shape[0])
    tmp = np.array(df)
    tmp = (-2 <= tmp)+(tmp<= 2)
    full = full * tmp
    return full

def make_full_path(root, df):
    df["RGB_Frontal_View"] = root + df['RGB_Frontal_View'].astype(str)
    return df

def load_resized(out_size, df):
    df["RGB_Frontal_View"] = pd.Series(np.array([cv2.resize(cv2.imread(path), out_size, interpolation="INTER_NEAREST") for path in df["RGB_Frontal_View"]]))
    return df

operations = dict({
            "Age" : norm,
            "Gender" : one_hot,
            "Weight": norm,
            "Height": norm,
            "Bodyfat:": norm,
            "Bodytemp": norm,
            "Sport-Last-Hour": convert_binary,
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
            "Gender" : [3],
            "Weight": [],
            "Height": [],
            "Bodyfat:": [0.05, 40.0],
            "Bodytemp": [],
            "Sport-Last-Hour": [],
            "Time-Since-Meal": [],
            "Tiredness": [9],
            "Radiation-Temp": [],
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
            "Ambient_Temperature" : [],
            "Ambient_Humidity" : [0.0, 100.0],
            "Emotion-ML" : [7],
            "Emotion-Self" : [6],
            "Label":[],
            "RGB_Frontal_View":[]})

def emotion2Id(x):
    i=0
    for key in ["Angry", "Fear", "Disgust", "Neutral", "Surprise", "Happy", "Sad"]:
        x.loc[(x["Emotion-ML"] == key), "Emotion-ML"] = i
        i += 1
    return x
    
def label2idx(label):
    idx = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
    try:
        return np.array(float(idx.index(label)))
    except:
        return np.array(label)

def get_change_rate(df, look_ahead_window=1000):
    df = df.values
    shifted = np.concatenate((df[look_ahead_window:], np.ones(look_ahead_window)*df[-1]))
    change_rate = shifted-df
    return change_rate

def feature_permutations(features, max_size=5):
    pms = []
    for i in range(2,max_size+1):
        pms.extend(list(p(features, i)))
    pms = list(set([tuple(sorted(x)) for x in pms]))
    return pms
            
        

if __name__ == "__main__":
    f = ["clothing","ambient_temp","ambient_hum"]
    
    tmp = feature_permutations(f)
    print(len(tmp))
    for i in range(len(tmp)):
         print(list(tmp[i]))