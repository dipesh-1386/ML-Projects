# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:45:23 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
df=pd.read_csv('car data.csv')
print(df.shape)
print(df.head())
print(df.describe())
##Finding missing values
print(df.isnull().sum())
sns.countplot(data=df,x='Transmission',hue='Selling_Price')
print(df['Transmission'].value_counts(), df['Fuel_Type'].value_counts())
df.replace({'Transmission':{'Manual':0,'Automatic':1},'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
X=df.drop(['Car_Name','Selling_Price'],axis=1)
y=df['Selling_Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)
##Training  model using Linear Regression
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred_train=reg.predict(X_train)
acc=r2_score(y_train, y_pred_train)
print("Training Accuracy Score= ",acc)
y_pred_test=reg.predict(X_test)
acc2=r2_score(y_test, y_pred_test)
print("Test Accuracy Score= ",acc2)
##Visualizing actual and predicted prices for training data , test data
plt.figure(figsize=(10,6))
plt.scatter(y_train,y_pred_train)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction")
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Car Price Prediction")
plt.show()

##Training using Lasso Regression
reg=Lasso()
reg.fit(X_train,y_train)
y_pred_train=reg.predict(X_train)
acc=r2_score(y_train, y_pred_train)
print("Training Accuracy Score= ",acc)
y_pred_test=reg.predict(X_test)
acc2=r2_score(y_test, y_pred_test)
print("Test Accuracy Score= ",acc2)
##Visualizing actual and predicted prices for training data , test data
plt.figure(figsize=(10,6))
plt.scatter(y_train,y_pred_train)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso - Car Price Prediction")
plt.show()
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred_test)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso - Car Price Prediction")
plt.show()