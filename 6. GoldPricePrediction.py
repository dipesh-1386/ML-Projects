# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:12:22 2023

@author: ds448
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
df=pd.read_csv('gld_price_data.csv')
print(df.shape)
print(df.head())
print(df.tail())
##Getting information about data
print(df.info())
##Finding missing values
print(df.isnull().sum())
##Getting statistical measures
print(df.describe())
corr=df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr,annot=True,cbar=True,square=True,fmt='.1f',annot_kws={'size':8})
plt.figure(figsize=(10,6))
sns.distplot(df['GLD'],color='green')
X=df.drop(['Date','GLD'],axis=1)
y=df['GLD']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
y_pred_train=rf.predict(X_train)
acc=r2_score(y_train, y_pred_train)
print("Training Accuracy Score= ",acc)
y_pred_test=rf.predict(X_test)
acc2=r2_score(y_test, y_pred_test)
print("Test Accuracy Score= ",acc2)
plt.figure(figsize=(10,6))
plt.scatter(y_train,y_pred_train)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Gold Price Prediction")
plt.show()