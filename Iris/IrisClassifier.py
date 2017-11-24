#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:31:58 2017

@author: pranavjain
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# get data from the dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, [1,2,3,4]].values
y = dataset.iloc[:, 5].values
# do this for labelEncoding since it is deprecated to directly use 1d arrays
y.reshape(-1,1)

# Label Encode the y set
# Iris-setosa -> 0
# Iris-versicolor -> 1
# Iris-virginica -> 2
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# predict our test results
y_pred = classifier.predict(X_test)

# to calculate the number of results that we got wrong
# eg. here we predicted only one wrong
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# predict for custom values
test = np.matrix('4 1 5 1')
test =sc.transform(test)
pred = classifier.predict(test)