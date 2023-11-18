# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:47:47 2023

@author: ds448
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 19:42:28 2023

@author: ds448
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('diabetes.csv')
print(df.head())
##Toatal rows,columns in dataset
print(df.shape)
##It tells mean of values of each column for each category
print(df.groupby('Outcome').mean())
##Describing the dataset
print(df.describe())
##Counting no. of values of each categorie(Rock, Mine)
print(df['Outcome'].value_counts())
##Dividing the dataset into dependent and independent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
##Splitting dataset into train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
##Feature scailing
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
##Training model with training data
from sklearn import svm
classifier=svm.SVC(C=5,kernel='linear')
classifier.fit(X_train_scaled,y_train)
##Predicting accuracy of train data
y_pred_train=classifier.predict(X_train_scaled)
from sklearn.metrics import accuracy_score
##Accuracy for training data
accuracy=accuracy_score(y_train, y_pred_train)
print("Accuracy for training data",accuracy)
##Predicting accuracy of train data
y_pred_test=classifier.predict(X_test_scaled)
##Accuracy for test data
accuracy=accuracy_score(y_test, y_pred_test)
print("Accuracy for test data",accuracy)

##Making predictive model
inputd=(10,168,74,0,0,38,0.537,34)
input_as_array=np.asarray(inputd)
inputac=input_as_array.reshape(1,-1)
inputac_scaled=scaler.transform(inputac)
print("Scaled values taken for prediction: ",inputac_scaled)
prediction=classifier.predict(inputac_scaled)
print("Predicted value is: ",prediction)
