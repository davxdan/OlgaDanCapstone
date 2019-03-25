# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:37:40 2019
@author: danie
"""
#%%
import numpy as np
import pandas as pd
pd.set_option("display.precision", 15)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from os import chdir, getcwd
chdir('C:\\Users\\danie\\Documents\\GitHub\\OlgaDanCapstone\\GPUProject')
#getcwd()
"""Silly way to load data"""
#data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float64}
#train = pd.read_csv('train.csv', dtype=data_type)
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
#%%
#acousticdata=acousticdata.reshape(-1, 1)
#timeToFailure=timeToFailure.reshape(-1, 1)
#%%
"""Get sample for Experimentation""""
#Rows,Columns
#idx = np.random.randint(629145480, size=100000)
#trainsample=train[idx,:]
#acousticData=trainsample[:,0].astype(np.int64)
#timeToFailure=trainsample[:,1].astype(np.float64)
#timeToFailure=np.ravel(timeToFailure)
acousticSample=train[0:100000,0]
timeToFailureSample=train[0:100000,1]
#%%
"""Data inspection"""
#len(train)
#train.ndim
#train.size
#train.dtype
#train.dtype.name
#len(acousticdata)
#acousticdata.ndim
#acousticdata.size
#acousticdata.dtype
#acousticdata.dtype.name
#b.astype(int) """Convert an array to a different type"""
#derp=sns.distplot(acousticSample)
#a4_dims = (11.7, 8.27)
#plt.subplots(figsize=(11.7, 8.27))
#sns.scatterplot(acousticSample,timeToFailureSample,size=acousticSample[0])
#sns.scatterplot(train[:,0],train[:,1])

plt.plot(train[:,0],train[:,1])
#axs[0].set_xlim(0, 2)
#axs[0].set_xlabel('time')
#axs[0].set_ylabel('s1 and s2')
#axs[0].grid(True)
plt.show()


acousticSample.mean
df = pd.DataFrame(train)
df.describe
df.dtypes
#%%
###########################Keras Experimientation##############################
#IMPORTANT TO SEE LINK BELOW FOR HOW TO ADD TIME TO DATA
###########################https://keras.io/preprocessing/sequence/############
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
#%%
'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

