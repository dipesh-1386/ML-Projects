# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:12:25 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df=pd.read_csv('creditcard.csv')
print(df.shape)
print(df.head())
print(df.tail())
##Getting information about data
print(df.info())
##Finding missing values
print(df.isnull().sum())
##Getting statistical measures
print(df.describe())
print(df['Class'].value_counts())
df.groupby('Class').mean()
##Seperating data with respect to 'Class'
legit=df[df.Class==0]
fraud=df[df.Class==1]
print(legit.shape)
print(fraud.shape)
print(legit.Amount.describe())
##Downsampling legit(with value '0') to get a balanced dataset
legit_sample=legit.sample(n=492)
##Concatenating 
ndf=pd.concat([fraud,legit_sample],axis=0)
ndf['Class'].value_counts()
ndf.groupby('Class').mean()
X=ndf.iloc[:,:-1]
y=ndf.iloc[:,-1]
##Spliting dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)
##Training model
reg=LogisticRegression()
reg.fit(X_train,y_train)
y_pred_train=reg.predict(X_train)
##Finding accuracy
acc=accuracy_score(y_train ,y_pred_train)
print("Training Accuracy score = ",acc)
y_pred_test=reg.predict(X_test)
acc2=accuracy_score(y_test ,y_pred_test)
print("Test Accuracy score = ",acc2)
