# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:37:40 2019
@author: danie
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
#%%
from os import chdir, getcwd
chdir('C:\\Users\\danie\\Documents\\GitHub\\OlgaDanCapstone\\GPUProject')
#getcwd()
#%%
"""Silly way to load data"""
#data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float64}
#train = pd.read_csv('train.csv', dtype=data_type)
#%%
"""Full Data Load and Save"""
def iter_loadtxt(filename, delimiter=',', skiprows=1, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data
#data = iter_loadtxt('train.csv')
#np.save('data', data)
train=np.load('data.npy')
#acousticData=data[:,0].astype(np.int64)
#timeToFailure=data[:,1].astype(np.float64)
#np.save('acousticdata', acousticdata)
#acousticdata=np.load('acousticdata.npy') 
#np.save('timeToFailure', timeToFailure)
#timeToFailure=np.load('timeToFailure.npy') 
#acousticdata=acousticdata.reshape(-1, 1)
#timeToFailure=timeToFailure.reshape(-1, 1)
#%%
"""Get sample for Experimentation""""
idx = np.random.randint(629145480, size=100000)
trainsample=train[idx,:]
acousticData=trainsample[:,0].astype(np.int64)
timeToFailure=trainsample[:,1].astype(np.float64)
timeToFailure=np.ravel(timeToFailure)
"""Data inspection"""
len(train)
train.ndim
train.size
train.dtype
train.dtype.name

len(acousticdata)
acousticdata.ndim
acousticdata.size
acousticdata.dtype
acousticdata.dtype.name
#b.astype(int) """Convert an array to a different type"""
plt.scatter(acousticData, timeToFailure)
#%%
"""Models"""
#https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),param_grid={'max_depth': range(3,7),
                                       'n_estimators': (10, 50, 100, 1000),},
                                       cv=5,scoring='neg_mean_squared_error',
                                       verbose=0,n_jobs=-1)
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], 
                                n_estimators=best_params["n_estimators"],
                                random_state=False, verbose=False)
# Perform K-Fold CV
    scores = cross_val_predict(rfr, X, y, cv=10, scoring='neg_mean_absolute_error')
    return scores
#%%
rfr_model(acousticData, timeToFailure)
