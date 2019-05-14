# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 09:07:26 2019

@author: danie
Copied from
https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import chdir, getcwd
chdir('C:\\Users\\danie\\Documents\\GitHub\\OlgaDanCapstone\\GPUProject')
#getcwd()

dataset_train = pd.read_csv('NSE-TATAGLOBAL.csv')
training_set = dataset_train.iloc[:, 1:2].values

#From previous experience with deep learning models, we know that we have to
#scale our data for optimal performance. In our case, we’ll use Scikit- Learn’s
#MinMaxScaler and scale our dataset to numbers between zero and one.
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#LSTMs expect our data to be in a specific format, usually a 3D array. We start 
#by creating data in 60 timesteps and converting it into an array using NumPy. 
#Next, we convert the data into a 3D dimension array with X_train samples, 60 
#timestamps, and one feature at each step.
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the LSTM
#In order to build the LSTM, we need to import a couple of modules from Keras:
#1. Sequential for initializing the neural network
#2. Dense for adding a densely connected neural network layer
#3. LSTM for adding the Long Short-Term Memory layer
#4. Dropout for adding dropout layers that prevent overfitting
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#We add the LSTM layer and later add a few Dropout layers to prevent 
#overfitting. We add the LSTM layer with the following arguments:
#
#1. 50 units which is the dimensionality of the output space
#2. return_sequences=True which determines whether to return the last output in
#the output sequence, or the full sequence 
#3. input_shape as the shape of our training set.
#
#When defining the Dropout layers, we specify 0.2, meaning that 20% of the 
#layers will be dropped. Thereafter, we add the Dense layer that specifies the 
#output of 1 unit. After this, we compile our model using the popular adam 
#optimizer and set the loss as the mean_squarred_error. This will compute the 
#mean of the squared errors. Next, we fit the model to run on 100 epochs with a 
#batch size of 32. Keep in mind that, depending on the specs of your computer, 
#this might take a few minutes to finish running.

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#Predicting Future Stock using the Test Set
#First we need to import the test set that we’ll use to make our predictions on
dataset_test = pd.read_csv('tatatest.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#In order to predict future stock prices we need to do a couple of things after 
#loading in the test set:
#1. Merge the training set and the test set on the 0 axis.
#2. Set the time step as 60 (as seen previously)
#3. Use MinMaxScaler to transform the new dataset
#4. Reshape the dataset as done previously
#After making the predictions we use inverse_transform to get back the stock 
#prices in normal readable format.
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Plotting the Results
#Finally, we use Matplotlib to visualize the result of the predicted stock 
#price and the real stock price.
plt.plot(real_stock_price, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()

#What's the purpose of adding a dense layer with a single unit at the end? The 
#last LSTM layer should output a single value (ie h for the last unit in the 
#layer), right?