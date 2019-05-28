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

data=np.load('data.npy')
acousticData=data[:,0].astype(np.int64)
timeToFailure=data[:,1].astype(np.float64)
del data
acousticData=acousticData.reshape(-1, 1)
timeToFailure=timeToFailure.reshape(-1, 1)
acousticData=acousticData[:627222016]
timeToFailure=timeToFailure[:627222016]

#np.save('acousticdata', acousticData)
#np.save('timeToFailure', timeToFailure)

#scale data
scaler = MinMaxScaler(feature_range = (0, 1))
acousticData = scaler.fit_transform(acousticData)  
timeToFailure = scaler.fit_transform(timeToFailure)
#testacousticData = scaler.fit_transform(testacousticData)  
#testtimeToFailure = scaler.fit_transform(testtimeToFailure)


testData = []
for i in range(856,8560,856):  
    testData.append(acousticData[i-856:i, 0])

testData = np.array(testData)  
testData = np.reshape(testData, (testData.shape[0], testData.shape[1], 1))


#whew made a cube 
trainData = []  
labels = []  
for i in range(856, 732737,856):  
    trainData.append(acousticData[i-856:i, 0])
    labels.append(timeToFailure[i, 0])
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












