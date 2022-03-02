
from PIL import Image
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from torch import *
import numpy as np



def rgb_loader(root, file):
    assert os.path.exists(file), "file not found: {}".format(root + file)
    return np.asarray(Image.open(file).convert('RGB'), dtype=np.uint8)

def one_hot(sample, classes=3):
    r = np.zeros((sample.size, classes))
    r[np.arange(sample.shape[0]), sample] = 1
    return r

def to_keypoint(sample):
    r = np.array([x.split('~') for x in sample]).astype(np.float32) 
    r[:,0] *= 1/1920
    r[:,1] *= 1/1080
    r[:,2] *= 1/5000
    
    return r

def to_tensor(img):
    if img.ndim == 3:
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
    elif img.ndim == 2:
        img = torch.from_numpy(img.copy())
    
    return img.float()

def norm(sample, min=None, max=None):
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
    q25, q75 = np.percentile(sample, 25), np.percentile(sample, 75)
    iqr = q75-q25 
    cutoff = iqr * cutoff_multiplier
    lower, upper = q25-cutoff, q75-cutoff
    mask = (sample < upper | sample > lower | sample > 0) 
    return mask


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