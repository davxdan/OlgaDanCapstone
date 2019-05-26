# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 14:37:40 2019
@author: danie
"""
#%%
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




"""Silly way to load data"""
#data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float64}
#train = pd.read_csv('train.csv', dtype=data_type)
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
#data=np.load('data.npy')
#print(min(data[:,0])) #-5515.0
#print(min(data[:,1])) #9.5503963166e-05
#dataFirst10000000=data[:10000000]
#np.savetxt("dataFirst10000000.csv", dataFirst10000000, delimiter=",")
#predict=np.load('seg_00a37e.npy')
#acousticData=data[:,0].astype(np.int64)
#timeToFailure=data[:,1].astype(np.float64)
#acousticData=acousticData.reshape(-1, 1)
#timeToFailure=timeToFailure.reshape(-1, 1)
#timeToFailure=timeToFailure[:155000]
#np.save('acousticdata', acousticData)
#np.save('timeToFailure', timeToFailure)

acousticData=np.load('acousticData.npy') 
#acousticData=acousticData[:155000]
timeToFailure=np.load('timeToFailure.npy') 

#Split Train from Time to Failure
trainTimeToFailure=timeToFailure[0:150000]
testTimeToFailure=timeToFailure[150000:155000]

#stack test data
total_data = np.vstack((trainTimeToFailure,testTimeToFailure,))

#Create test input (Get the last 5000 records from total data)
test_inputs = total_data[len(total_data) - len(testTimeToFailure):]

#reshape data
trainTimeToFailure=trainTimeToFailure.reshape(-1, 1)
testTimeToFailure=testTimeToFailure.reshape(-1,1)
acousticData=acousticData.reshape(-1,1)

#scale data
scaler = MinMaxScaler(feature_range = (0, 1))
trainTimeToFailureScaled = scaler.fit_transform(trainTimeToFailure)  
testTimeToFailureScaled = scaler.fit_transform(testTimeToFailure)
acousticDataScaled = scaler.fit_transform(acousticData)




#for i in range(60, 1259):  
#    features_set.append(train_data_scaled[i-60:i, 0])
#    labels.append(train_data_scaled[i, 0])
#
#features_set, labels = np.array(features_set), np.array(labels)  
#features_set.shape


#features_set = []  
#labels = []  
#for i in range(0,4194):  
#    features_set.append(acousticDataScaled[i:i+150000, 0])
#    labels.append(trainTimeToFailureScaled[i, 0])

trainfeatures_set = []  
labels = []  
for i in range(150000,629100000,150000):  
    trainfeatures_set.append(acousticDataScaled[i-150000:i, 0])
    labels.append(acousticDataScaled[i, 0])

trainfeatures_set, labels = np.array(trainfeatures_set), np.array(labels)  
print(trainfeatures_set.shape)


test_features = []  

for i in range(150000, 4500000, 150000):  
    test_features.append(acousticDataScaled[i-150000:i, 0])


test_features = np.array(test_features)  


trainfeatures_set = np.reshape(trainfeatures_set, (trainfeatures_set.shape[0], trainfeatures_set.shape[1], 1))  

test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))  



model = Sequential()  
#Note that 50 is the number of hidden units, return sequences is required to add subsequent LSTM layers
model.add(LSTM(units=500, return_sequences=True, input_shape=(trainfeatures_set.shape[1], 1)))  
model.add(Dropout(0.2))  
model.add(LSTM(units=500, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=500, return_sequences=True))  
model.add(Dropout(0.2))
model.add(LSTM(units=500))  
model.add(Dropout(0.2))  
model.add(Dense(units = 1))  

#Since this is a time-based regression problem, mean_squared_error is chosen for our loss function
#adam will automatically update the learning rate for us.
model.compile(optimizer = 'adam', loss = 'mean_squared_error')  

#We fit the LSTM model with the input features, labels, 100 epochs, and a batch size of 32
#Note that the batch size is very important in an LSTM. A larger batch size might not be appropriate.
#Experimentation is recommended to find a reasonable batch size for your data.

#This will take a long time to fit... For me, it was about 40 minutes, but I didn't time it precisely.
model.fit(trainfeatures_set, labels, epochs = 10, batch_size = 500)  

#Then, then predictions with the scaled test features occur quickly
predictions = model.predict(test_features)  

#Then, predictions should be scaled back to the original scale.
predictions = scaler.inverse_transform(predictions)  


#%%
"""Big Data inspection"""
#print(max(predict[:,0])) #162.0
#print(min(predict[:,0])) #-138
#print(max(train[:,1])) #16.1074
#print(min(train[:,1])) #9.5503......
#acousticData.max(axis=0)#5444
#acousticData.min(axis=0)#-5515
#fig, ax = plt.subplots(figsize=(11, 8.5))
#ax.plot(train[:,1],train[:,0])
#ax.set(xlabel='Time to Failure(seconds)', ylabel='Siesmic Signals',
#       title='LANL Siesmic Signals by Time To Failure: 629,145,480 Observations')
#ax.grid()
#fig.savefig("allDataDefaultPlot.png")
#plt.show()
#
#fig, ax = plt.subplots(figsize=(11, 8.5))
#sns.distplot(train[:,0],axlabel='Siesmic Signals:629,145,480 Observations',label='LANL Siesmic Signals Distribution:629,145,480 Observations')
#fig.savefig("acousticRand60000DistPlot.png")
#%%
"""Sample Data Inspection"""
#Rows,Columns
#idx = np.random.randint(629145480, size=60000)
#trainsample=train[idx,:]

#Split the trainsample into 2 arrays
#acousticData = np.delete(trainsample, np.s_[1],axis=1)
#acousticData.shape
#timeToFailure = np.delete(trainsample, np.s_[0],axis=1)
#timeToFailure.shape
#timeToFailure=np.ravel(timeToFailure)
#Enormous outliers presumably siesmic failure or major slip.
#%%


#sns.scatterplot(acousticSample,timeToFailureSample,size=acousticSample[0])
#sns.scatterplot(train[:,0],train[:,1])
#%%



#%%
###########################Keras Experimientation##############################
#IMPORTANT TO SEE LINKS BELOW FOR HOW TO ADD TIME TO DATA 
#I may need to zip the data so the acoustic signal is folowed by the time to 
#failure. This can make it look like natrual language
#https://adventuresinmachinelearning.com/recurrent-neural-networks-lstm-tutorial-tensorflow/
###########################https://keras.io/preprocessing/sequence/############
"""
Created on Sat Mar 30 09:49:24 2019

@author: Chris
"""

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

#For background on LSTMs, read the following: http://colah.github.io/posts/2015-08-Understanding-LSTMs/

#I modifed the example at https://stackabuse.com/time-series-analysis-with-lstm-using-pythons-keras-library/
# and I used NumPy instead of Pandas.

#You could easily go and get your own stock data from Yahoo finance to do the same below.
#Or, you could download the data as I did, from the commented links below for Apple.

#Download training data from : 
# https://finance.yahoo.com/quote/AAPL/history?period1=1357020000&period2=1514700000&interval=1d&filter=history&frequency=1d

#Download test data from :
# https://finance.yahoo.com/quote/AAPL/history?period1=1514786400&period2=1517378400&interval=1d&filter=history&frequency=1d

#Import data
train_data = np.genfromtxt('data\AAPL_train_data.csv', delimiter=',')
test_data = np.genfromtxt('data\AAPL_test_data.csv', delimiter=',')

#examine data
np.set_printoptions(threshold=sys.maxsize)
train_data.shape
test_data.shape

#delete first row that was a header row
train_data = np.delete(train_data, 0, 0)
test_data = np.delete(test_data, 0, 0)

#delete first column that was a date column 
train_data = np.delete(train_data, 0, 1)
test_data = np.delete(test_data, 0, 1)

#examine data
train_data.shape
test_data.shape

#Only retain open price column
train_data = np.delete(train_data, np.s_[1:6],axis=1)
test_data = np.delete(test_data, np.s_[1:6],axis=1)

train_data.shape
test_data.shape

#stack data together for next step
total_data = np.vstack((train_data,test_data,))

total_data.shape
total_data

#create test inputs
#You will see why we are doing this below.
#Basically, we are getting the 60 prior values for each test_data value. There are 21 test values
test_inputs = total_data[len(total_data) - len(test_data) - 60:]
test_inputs.shape

#Data normalization
#Recall that scaling the data is important with in neural networks to help prevent overfitting.
#first create the scaler
scaler = MinMaxScaler(feature_range = (0, 1))

#reshape data
train_data_reshaped = train_data.reshape(-1, 1)
test_inputs_reshaped = test_inputs.reshape(-1, 1)

#scale data
train_data_scaled = scaler.fit_transform(train_data_reshaped)  
test_inputs_scaled = scaler.fit_transform(test_inputs_reshaped)

train_data_scaled.shape

#preprocess training data
#We are predicting a value at time T, based on the data from days T-N, and N can be any number.
#We are predicting the opening stock price based on opening stock prices for the past 60 days
#Therefore, starting at the 61st record, we are getting the 60 prior opening stock prices as 
#input features for each row. Therefore, as an example, for the 61st record, the prior 60 records are
#stored in features set, and the 61st record value is stored in labels for each given row.
features_set = []  
labels = []  
for i in range(60, 1259):  
    features_set.append(train_data_scaled[i-60:i, 0])
    labels.append(train_data_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)  
features_set.shape

#preprocess test data
#given what we need for input features, for testing, we also need the 60 prior values to the first
#value in the testing dataset.
test_features = []  
for i in range(60, 81):  
    test_features.append(test_inputs_scaled[i-60:i, 0])

test_features = np.array(test_features)  
test_features.shape


features_set.shape
labels.shape

#We need to reshape to the input format that LSTM's require, so we are adding 1 channel as a 3rd dimension
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))  
features_set.shape


test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))  
test_features.shape



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
model.fit(features_set, labels, epochs = 10, batch_size = 5000)  

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
model.fit(features_set, labels, epochs = 100, batch_size = 5000)  

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