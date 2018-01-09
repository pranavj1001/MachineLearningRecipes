#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 23:19:52 2017

@author: pranavjain

This model classifies a mushroom as edible or poisonous
Attribute Information:
For y (dependent variables)
(classes: edible=e, poisonous=p) after labelEncoding (edible=0, poisonous=1)

For X (independent variables)
cap-shape:
    bell=b,conical=c, convex=x, flat=f, knobbed=k, sunken=s
cap-surface:
    fibrous=f, grooves=g, scaly=y, smooth=s
cap-color:
    brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
bruises:
    bruises=t, no=f
odor:
    almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
gill-attachment:
    attached=a, descending=d, free=f, notched=n
gill-spacing:
    close=c, crowded=w, distant=d
gill-size:
    broad=b, narrow=n
gill-color:
    black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
stalk-shape:
    enlarging=e, tapering=t
stalk-root:
    bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
stalk-surface-above-ring:
    fibrous=f, scaly=y, silky=k, smooth=s
stalk-surface-below-ring:
    fibrous=f, scaly=y, silky=k, smooth=s
stalk-color-above-ring:
    brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
stalk-color-below-ring:
    brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y
veil-type:
    partial=p, universal=u
veil-color:
    brown=n, orange=o, white=w, yellow=y
ring-number:
    none=n, one=o, two=t
ring-type:
    cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
spore-print-color:
    black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y
population:
    abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
habitat:
    grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d

"""

# import libraries
import pandas as pd

# get data from the dataset
dataset = pd.read_csv('mushrooms.csv')
X = dataset.iloc[:, 1:23]
y = dataset.iloc[:, 0].values

# Label Encode y and X
# for eg. in y
# Poisonous -> 1
# Edible -> 0
from sklearn.preprocessing import LabelEncoder
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
X = X.apply(LabelEncoder().fit_transform)
X = X.iloc[:, :].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# predict our test results
y_pred = classifier.predict(X_test)

# to calculate the number of results that we got wrong
# eg. here we predicted 747 / 813 i.e. an accuracy of 91.88%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
