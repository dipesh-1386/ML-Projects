# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:42:28 2023

@author: ds448
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('sonar.csv',header=None)
print(df.head())
##Toatal rows,columns in dataset
print(df.shape)
##It tells which colun
print(df.groupby(60).mean())
##Describing the dataset
print(df.describe())
##Counting no. of values of each categories(Rock, Mine)
print(df[60].value_counts())
##Dividing the dataset into dependent and independent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
##Splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)
##Training model with training data
from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(X_train,y_train)
##Predicting accuracy of train data
y_pred_train=reg.predict(X_train)
from sklearn.metrics import accuracy_score
##Accuracy for training data
accuracy=accuracy_score(y_train, y_pred_train)
print("Accuracy for training data",accuracy)
##Predicting accuracy of train data
y_pred_test=reg.predict(X_test)
##Accuracy for test data
accuracy=accuracy_score(y_test, y_pred_test)
print("Accuracy for test data",accuracy)

##Making predictive model
inputd=(0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)
input_as_array=np.asarray(inputd)
inputac=input_as_array.reshape(1,-1)
print("Predicted value is: ",reg.predict(inputac))
