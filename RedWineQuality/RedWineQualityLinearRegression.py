#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:49:57 2018

@author: pranavjain
"""

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality-red.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

# Deprecation warnings call for reshaping of single feature arrays with reshape(-1,1)
y = y.reshape(-1,1)
# avoid DataConversionError
y = y.astype(float)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_test = sc_y.transform(y_test)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# consider p-value as 10
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((1599, 1)).astype(float), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# drop 'density'
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# drop 'fixed acidity'
X_opt = X[:, [0, 2, 3, 4, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# drop 'residual sugar'
X_opt = X[:, [0, 2, 3, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# drop 'critic acid'
X_opt = X[:, [0, 2, 5, 6, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# drop 'free sulphur dioxide'
X_opt = X[:, [0, 2, 5, 7, 9, 10, 11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
# hence the optimal model is now ready