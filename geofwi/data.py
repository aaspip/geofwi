# dataloader
import torch
import torch.utils.data as data
import re
import os
import os.path
import numpy as np
import scipy.io as sio
#import matlab.engine
#eng = matlab.engine.start_matlab()

from IPython.core import debugger
debug = debugger.Pdb().set_trace

import sys

IMG_EXTENSIONS = ['.mat']


class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(*data)
        return data


class JointRandomFlip(object):
    # Data and model flipling for data augmentation
    def __init__(self, rand=True):
        self.rand = rand

    def __call__(self, observe, geology):
        random = torch.rand(1)
        if self.rand and random < 0.5:
            geology = geology.flip(2)
            observe = observe.flip([0,1])
        return observe, geology


class JointNormalize(object):
    def __init__(self, norm=True):
        self.norm = norm

    def __call__(self, observe, geology):
        if self.norm:
            
            #geology = (geology-1500.0)/2500.0
            #observe = 2*(observe+8)/(8+15)-1
            
            geology = (geology-1500.0)/3000.0
            observe = 2*(observe+40)/(90)-1
            
        return observe, geology    


def is_mat_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def mat_loader(path):
    try:
        data = sio.loadmat(path)
    except Exception:
        print(path)
        print('Error:', Exception)
    #finally:
    return data


class DatasetFolder(data.Dataset):
    def __init__(self, root, flip=True, norm=True):
        self.flip = flip
        self.root = root
        self.norm = norm
        self.data = data
        self.obs_names = self.get_data(self.root+'/data')
        print(len(self.obs_names))
        self.geo_names = self.get_con(self.obs_names)
        print(len(self.geo_names))
        
    def get_data(self, path):
        """
        groups = [
            d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
        ]
        """
        groups = [
            path
        ]
        data_list = []
       
        for i in sorted(groups):
            
            group_path = os.path.join(path, i)
            
            
            if not os.path.isdir(group_path):
                continue
            data = [
                os.path.join(group_path, d)
                for d in os.listdir(group_path) if is_mat_file(d)
                ]
            
            data.sort(key=lambda x: int(re.findall(r'(\d+)', x)[-1]))
            #data_list += data[0:2000]
            data_list += data[0:2500]
        return data_list

    def get_con(self, data_list):
        con_list = [i.replace("data", "model") for i in data_list]
        return con_list

    def __getitem__(self, index):
        #load data
        geology_path = self.geo_names[index]
        observe_path = self.obs_names[index]
        geology = mat_loader(geology_path)['param_sum']
        try:
            observe = mat_loader(observe_path)['ob_data']
        except Exception:
            observe = mat_loader(observe_path)['simple_data']
            
        
        observe = torch.from_numpy(observe)
        geology = torch.from_numpy(geology)
        
        
        #print(observe.shape)
        #print(geology.shape)
        
        geology = geology.unsqueeze(0)
        
       
       
            
        transform = JointCompose([JointRandomFlip(self.flip), JointNormalize(self.norm)])
        observe, geology = transform([observe, geology])
        observe=observe[:,::2,::2] 
        #print(geology.shape)
        #sys.exit()
        # geology size [100,100]
        # observe size [20,32,1000]

        return observe, geology, observe_path, geology_path

    def __len__(self):
        return len(self.geo_names)
