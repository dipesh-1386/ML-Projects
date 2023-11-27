# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:34:00 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df=pd.read_csv('insurance.csv')
print(df.shape)
print(df.head())
print(df.tail())
##Getting information about data
print(df.info())
##Finding missing values
print(df.isnull().sum())
##Getting statistical measures
print(df.describe())
print("Charges info ------>>>>",df.charges.describe())
print("Value counts---->>>> ",df['sex'].value_counts())
print(df['smoker'].value_counts())
print(df['region'].value_counts())
df.replace({'sex':{'male':0,'female':1},'smoker':{'no':0,'yes':1},'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)
corr=df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True,cbar=True,annot_kws={'size':8},fmt='.1f')
plt.figure(figsize=(10,6))
sns.set()
sns.distplot(df['age'],color='green')
plt.figure(figsize=(10,6))
sns.countplot(x='children',data=df)
plt.figure(figsize=(10,6))
sns.set()
sns.distplot(df['charges'],color='blue')
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred_train=reg.predict(X_train)
##Finding r2score
acc=r2_score(y_train ,y_pred_train)
print("Training Accuracy score = ",acc)
y_pred_test=reg.predict(X_test)
acc2=r2_score(y_test ,y_pred_test)
print("Test Accuracy score = ",acc2)