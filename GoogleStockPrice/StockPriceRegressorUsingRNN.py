#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:15:23 2018

@author: pranavjain

This program trains on Google stock prices and then predicts Google stock prices for the month of January, 2017.

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Making inputs and outputs
X_train = training_set_scaled[0:1257]
y_train = training_set_scaled[1:1258]

# reshaping X_train
X_train = np.reshape(X_train, (1257, 1, 1))

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding first layer of LSTM
regressor.add(LSTM(units = 50, activation = 'sigmoid', input_shape = (None, 1), return_sequences = True))

# Adding second layer of LSTM
regressor.add(LSTM(units = 50, activation = 'sigmoid', return_sequences = True))

# Adding third layer of LSTM
regressor.add(LSTM(units = 50, activation = 'sigmoid'))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 32)

# Predict the stock price of Google from the year 2017 (first month)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
jan_2017_stock_price = dataset_test.iloc[:, 1:2].values
X_test = jan_2017_stock_price

# Feature Scale the X_test
X_test = sc.transform(X_test)

# Making X_test a 3D array
X_test = np.reshape(X_test, (20, 1, 1))

# Predicting the results
y_pred = regressor.predict(X_test)

# Inverse Transform the predictions
y_pred = sc.inverse_transform(y_pred)

# Visualising the results
plt.plot(jan_2017_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
