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
