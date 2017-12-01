#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:25:17 2017

@author: pranavjain

This model predicts whether a restaurant review is good or bad.

"""

# Importing the libraries
import pandas as pd

# Importing the dataset
# logic to read a tsv file
# quoting = 3 -> ignore double quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# stem the words eg. consider 'love' instead of 'loved'
# ignore words like 'the', 'on', 'a', etc which don't have much meaning
# consider lower case letters
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # consider only letters from reviews
    review = review.lower() # covert all letters to lowvercase
    review = review.split() # split the string to a list of words
    ps = PorterStemmer() # object for our stemmer
    # remove words like 'this', 'the', etc words which are present in nltk stopwords
    # and consider only the stem of the word
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # convert the list of words to a string
    # eg. ['wow', 'love', 'place'] -> wow love place
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=40, criterion='gini', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
# here the accuracy is (87 + 61) / 200 -> 74%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
