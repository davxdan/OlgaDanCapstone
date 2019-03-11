# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:37:40 2019

@author: danie
"""

#%%
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#%%
from os import chdir, getcwd
chdir('C:\\Users\\danie\\Documents\\GitHub\\OlgaDanCapstone\\GPUProject')
#getcwd()
#%%
#traindata = pd.read_csv('train.csv')
#%%
#traindata=np.genfromtxt("train.csv", delimiter=',')
#np.loadtxt('train.csv',dtype={'names': ('acoustic_data', 'time_to_failure'),'formats': (np.int,np.float64)},delimiter=',', skiprows=1)
#%%
if torch.cuda.is_available():

    # creates a LongTensor and transfers it
    # to GPU as torch.cuda.LongTensor
    a = torch.full((10,), 3, device=torch.device("cuda"))
    print(type(a))
    b = a.to(torch.device("cpu"))
    # transfers it to CPU, back to
    # being a torch.LongTensor
print("I am the law")
#print(traindata)
#len(a) """Length of array"""
#b.ndim """Number of array dimensions"""
#e.size """Number of array elements"""
#b.dtype """Data type of array elements"""
#b.dtype.name """Name of data type"""
#b.astype(int) """Convert an array to a different type"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import glob
#import cv2
import random
import os.path
from datetime import datetime
import pandas as pd
np.random.seed(2016)
random.seed(2016)
#%%
conf = dict()
# Shape of image for CNN (Larger the better, but you need to increase CNN as well)
conf['image_shape'] = (32,32)
#%%
print(str(datetime.now()))
filepaths = []
filepaths.append('../input/train/Type_1/')
filepaths.append('../input/train/Type_2/')
filepaths.append('../input/train/Type_3/')
filepaths.append('../input/test/')
print(str(datetime.now()))
#%%
print(str(datetime.now()))
allFiles = []
for i, filepath in enumerate(filepaths):
    files = glob.glob(filepath + '*.jpg')
    allFiles = allFiles + files

print(str(datetime.now()))