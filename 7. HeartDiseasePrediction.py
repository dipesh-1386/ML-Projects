# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:18:12 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('heart_disease_data.csv')
print(df.shape)
print(df.head())
print(df.tail())
##Getting information about data
print(df.info())
##Finding missing values
print(df.isnull().sum())
##Getting statistical measures
print(df.describe())
plt.figure(figsize=(10,6))
sns.countplot(x='cp',hue='target',data=df)
plt.figure(figsize=(10,6))
sns.countplot(x='slope',hue='target',data=df)
plt.figure(figsize=(10,6))
sns.countplot(x='restecg',hue='target',data=df)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
reg=LogisticRegression()
reg.fit(X_train,y_train)
y_pred_train=reg.predict(X_train)
acc=accuracy_score(y_train, y_pred_train)
print("Training Accuracy Score= ",acc)
y_pred_test=reg.predict(X_test)
acc2=accuracy_score(y_test, y_pred_test)
print("Test Accuracy Score= ",acc2)
##Making predictive model
input_data=(64,1,3,170,227,0,0,155,0,0.6,1,0,3)
input_data_arr=np.asanyarray(input_data)
input_data_arr=input_data_arr.reshape(1,-1)
print("Prediction for inputted data ->",reg.predict(input_data_arr))
