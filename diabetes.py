#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 21:32:26 2018

@author: sachin1006
"""

# Import the packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#The dataset is available on github on this link.
#Import the dataset.
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
dataset=pd.read_csv(url)
print(dataset)
X=dataset.iloc[:,0:7].values
Y=dataset.iloc[:,8].values

#Splitting the dataset into training set and test set for furthur training of the models.
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0)



#SVM  to train the model and predict the results.
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#Naive bayes algorithm to train the model and predict the result.
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Formation of confusion matrix to visualize your predicted result data.
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#percentage accuracy.
result=np.mean(y_test==y_pred)*100

# Applying k-Fold Cross Validation to check the best suitable parameter values to be used.
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracy=accuracies.mean()*100
accuracies.std()


#NOTE:
#SVM predicts the result with an accuracy of 78.35%.
#Naive Bayes predicts the result with an accuracy of 80%.

################################END#####################################