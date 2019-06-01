# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout  
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from os import chdir, getcwd
chdir('C:\\Users\\danie\\Documents\\GitHub\\OlgaDanCapstone\\GPUProject')
import seaborn as sns
sns.set(color_codes=True)
#getcwd()
np.set_printoptions(threshold=sys.maxsize)
pd.set_option("display.precision", 15)
"""Full Data Load and Save"""
#def iter_loadtxt(filename, delimiter=',', skiprows=1, dtype=float):
#    def iter_func():
#        with open(filename, 'r') as infile:
#            for _ in range(skiprows):
#                next(infile)
#            for line in infile:
#                line = line.rstrip().split(delimiter)
#                for item in line:
#                    yield dtype(item)
#        iter_loadtxt.rowlength = len(line)
#
#    data = np.fromiter(iter_func(), dtype=dtype)
#    data = data.reshape((-1, iter_loadtxt.rowlength))
#    return data
#data = iter_loadtxt('train.csv')
#data = iter_loadtxt('seg_00a37e.csv')
#np.save('seg_00a37e', data)
data=np.load('data.npy')
trainTimeToFailure=data[:75000,1].astype(np.float64)
trainTimeToFailure=trainTimeToFailure.reshape(-1, 1)

testTimeToFailure=data[75000:150000,1].astype(np.float64)
testTimeToFailure=testTimeToFailure.reshape(-1, 1)
del data

#np.save('acousticdata', acousticData)
#np.save('timeToFailure', timeToFailure)

#scale data
scaler = MinMaxScaler(feature_range = (0, 1))
trainTimeToFailure = scaler.fit_transform(trainTimeToFailure)
testTimeToFailure = scaler.fit_transform(testTimeToFailure)

total_data = np.vstack((trainTimeToFailure,testTimeToFailure,))

test_inputs = total_data[len(total_data) - len(testTimeToFailure) - 10000:]

features_set = []  
labels = []  
for i in range(10000, 75000):  
    features_set.append(trainTimeToFailure[i-10000:i, 0])
    labels.append(trainTimeToFailure[i, 0])
features_set, labels = np.array(features_set), np.array(labels)  
del trainTimeToFailure

test_features = []  
for i in range(10000, 75000):  
    test_features.append(test_inputs[i-10000:i, 0])
test_features = np.array(test_features) 

    
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))  


del i
del testTimeToFailure
del total_data
















#32 batch size
model = Sequential()  
#Note that 50 is the number of hidden units, return sequences is required to add subsequent LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  

#Since this is a time-based regression problem, mean_squared_error is chosen for our loss function
#adam will automatically update the learning rate for us.
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

#We fit the LSTM model with the input features, labels, 100 epochs, and a batch size of 32
#Note that the batch size is very important in an LSTM. A larger batch size might not be appropriate.
#Experimentation is recommended to find a reasonable batch size for your data.

#This will take a long time to fit... For me, it was about 40 minutes, but I didn't time it precisely.
model.fit(features_set, labels, epochs = 2, batch_size = 5000)  

#Then, then predictions with the scaled test features occur quickly
predictions = model.predict(test_features)  

#Then, predictions should be scaled back to the original scale.
predictions = scaler.inverse_transform(predictions)  


#To show the differences between different batch sizes, you can also try a batch size of 100
#This will run a bit quicker.

#100 batch size
model = Sequential()  
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))  
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=50))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  


#This is the same as with the LSTM with batch size 32
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

#Here we change the batch size from 32 to 100.
model.fit(features_set, labels, epochs = 10, batch_size = 5000)  

#name these predictions a differently to plot both them and the previous against the actuals
predictions2 = model.predict(test_features)  
predictions2 = scaler.inverse_transform(predictions2)  


#plot the results.
#Note that these predictions are for the month of January 2018 and they are not labeled correctly.
#Given a bit more effort, we could get dates to show from the original csv.
plt.figure(figsize=(10,6))  
plt.plot(test_data, color='blue', label='Actual Apple Stock Price')  
plt.plot(predictions , color='red', label='Predicted Apple Stock Price (batch=32')  
plt.plot(predictions2 , color='green', label='Predicted Apple Stock Price (batch=100')  
plt.title('Apple Stock Price Prediction')  
plt.xlabel('Date')  
plt.ylabel('Apple Stock Price')  
plt.legend()  

#Save the figure and show the figure.
plt.savefig("AAPL LSTM actuals vs pred", bbox_inches="tight")
plt.show()  








