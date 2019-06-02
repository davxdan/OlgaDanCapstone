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
trainAcousticData=data[:10000,0].astype(np.int64)
trainTimeToFailure=data[:10000,1].astype(np.float64)
trainAcousticData=trainAcousticData.reshape(-1, 1)
trainTimeToFailure=trainTimeToFailure.reshape(-1, 1)
testAcousticData=data[10000:20000,0].astype(np.int64)
testTimeToFailure=data[10000:20000,1].astype(np.float64)
testAcousticData=testAcousticData.reshape(-1, 1)
testTimeToFailure=testTimeToFailure.reshape(-1, 1)
del data

#np.save('acousticdata', acousticData)
#np.save('timeToFailure', timeToFailure)

#scale data
#scaler = MinMaxScaler(feature_range = (0, 1))
#trainAcousticData = scaler.fit_transform(trainAcousticData)  
#trainTimeToFailure = scaler.fit_transform(trainTimeToFailure)
#testAcousticData = scaler.fit_transform(testAcousticData)  
#testTimeToFailure = scaler.fit_transform(testTimeToFailure)


testData = []
for i in range(500,10000,20):  
    testData.append(testAcousticData[i-500:i, 0])

testData = np.array(testData)  
testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1))


#whew made a cube 
trainData = []  
labels = []  
for i in range(1,10000,1):  
    trainData.append(trainAcousticData[i-1:i, 0])
    labels.append(trainTimeToFailure[i-1:i, 0])
trainData, labels = np.array(trainData), np.array(labels)  
trainData = np.reshape(trainData, (trainData.shape[0], trainData.shape[1], 1))

#model.fit(x_train, y_train, batch_size=16, epochs=10)
#score = model.evaluate(x_test, y_test, batch_size=16)
model = Sequential()  
model.add(LSTM(units=50, return_sequences=True, input_shape=(trainData.shape[1], 1)))  
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
model.fit(trainData, labels, epochs = 2, batch_size = 5000)  

predictions = model.predict(testData)
predictions = scaler.inverse_transform(predictions)


plt.figure()  
plt.plot(trainData, color='blue', label='Actual Apple Stock Price')  
plt.plot(predictions , color='red', label='Predicted Apple Stock Price (batch=32')  

plt.title('Apple Stock Price Prediction')  
plt.xlabel('Date')  
plt.ylabel('Apple Stock Price')  
plt.legend()  

#Save the figure and show the figure.

plt.show()  












