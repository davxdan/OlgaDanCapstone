# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 22:21:29 2019

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
cwd = os.getcwd()
print(cwd)
#%%
#traindata = pd.read_csv('train.csv')
#%%
traindata=np.genfromtxt("train.csv", delimiter=',')
#%%
print("I am the law")
print(traindata)
#len(a) """Length of array"""
#b.ndim """Number of array dimensions"""
#e.size """Number of array elements"""
#b.dtype """Data type of array elements"""
#b.dtype.name """Name of data type"""
#b.astype(int) """Convert an array to a different type"""