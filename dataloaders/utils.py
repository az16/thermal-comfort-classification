
from PIL import Image
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
import scipy.ndimage.interpolation as itpl
import torch
import numpy as np
import os


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


    
def rgb_loader(root, file):
    # does not work with anything other than path parameter
    #TODO: implement sequence loading in case of RNN dataloading
    assert os.path.exists(file), "file not found: {}".format(root + file)
    return np.asarray(Image.open(file).convert('RGB'), dtype=np.uint8)

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
    if img.ndim == 3:
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
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

# TODO: cite pythermal creators
def pmv(p_vars, t_a, rh_a):
    tdb = t_a 
    tr = p_vars[:,0]
    icl = p_vars[:1]
    v = p_vars[:,2]
    rh = rh_a * 100.0
    met = p_vars[:,3]
    
    vr = v_relative(v, met)
    clo = clo_dynamic(icl, met)
    
    return np.array([pmv_ppd(tdb[x], tr[x], vr[x], rh[x], met[x], clo[x])["pmv"] for x in range(p_vars.shape[0])])

def rotate(angle, img, reshape=False, prefilter=False, order=0):
    return itpl.rotate(img, angle, reshape, prefilter, order)

def center_crop(img, output_size):
    h = img.shape[0]
    w = img.shape[1]
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i+h, j:j+w, :]

def horizontal_flip(img, do_flip):
    if do_flip: return np.fliplr(img)
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

def convert_binary(x):
    
    return (x == 1)

def place_holder(x):
    return x

def noise(shape, mean = 0, std = 0.3):
    return np.random.normal(mean, std, shape)

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
            "Radiation-Temp": [10,40],
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
            "Label":[]})

def label2idx(label):
    idx = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
    return np.array(float(idx.index(label)))

    