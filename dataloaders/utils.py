
from PIL import Image
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
import scipy.ndimage.interpolation as itpl
import torch
import numpy as np
import os

    
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
    if not min is None and not max:
        max = np.max(sample)
    elif not max is None and not min:
        min = np.min(sample)
    else:
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
    mask_z = (sample > 0) 
    mask = np.logical_and(mask_u, mask_l)
    return np.logical_and(mask, mask_z)

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

def label2idx(label):
    idx = [-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0]
    return np.array(float(idx.index(label)))

    