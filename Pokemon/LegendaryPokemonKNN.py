#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:08:26 2017

@author: pranavjain

This model classifies a pokemon as legendary or not-legendary.
Required Data to predict Hit Points, Attack Points, Defence Points, Special Attack Points, Special Defence Points, Speed Points

"""

# import libraries
import pandas as pd
import numpy as np

# get data from the dataset
dataset = pd.read_csv('Pokemon.csv')
X = dataset.iloc[:, [5,6,7,8,9,10]].values
y = dataset.iloc[:, 12].values
# do this for labelEncoding since it is deprecated to directly use 1d arrays
y.reshape(-1, 1)

# Label Encode the y set
# False -> 0
# True -> 1
from sklearn.preprocessing import LabelEncoder
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
# eg. here we predicted only 15 wrong out of 200
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# predict for custom values
# Arguments(Hit Points, Attack Points, Defence Points, Special Attack Points, Special Defence Points, Speed Points)
test = np.matrix('106 190 100 154 100 130')
test =sc.transform(test)
pred = classifier.predict(test)
