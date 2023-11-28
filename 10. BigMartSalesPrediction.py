# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 15:27:56 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
df=pd.read_csv('Bigmartsales.csv')
print(df.shape)
print(df.head())
print(df.tail())
##Getting information about data
print(df.info())
##Finding missing values
print(df.isnull().sum())
##Getting statistical measures
print(df.describe())
df.isnull().sum()
##mean of Item_weight
mean=df['Item_Weight'].mean()
print(mean)
##mode of outlet size
mode_outlet=df.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0]))
print(mode_outlet)
df['Item_Weight'].fillna(mean,inplace=True)
missing_values=df['Outlet_Size'].isnull()
df.loc[missing_values,'Outlet_Size']=df.loc[missing_values,'Outlet_Type'].apply(lambda x: mode_outlet[x])
df.isnull().sum()
sns.set()
plt.figure(figsize=(10,6))
sns.distplot(df['Item_Weight'])
sns.set()
plt.figure(figsize=(10,6))
sns.distplot(df['Item_Visibility'])
plt.figure(figsize=(10,6))
sns.distplot(df['Item_Outlet_Sales'])
plt.figure(figsize=(10,6))
sns.distplot(df['Item_MRP'])
plt.figure(figsize=(10,6))
sns.countplot(x='Outlet_Establishment_Year',data=df)
plt.show()
plt.figure(figsize=(25,6))
sns.countplot(x='Item_Type',data=df)
plt.show()
plt.figure(figsize=(10,6))
sns.countplot(x='Item_Fat_Content',data=df)
plt.show()
plt.figure(figsize=(10,6))
sns.countplot(x='Outlet_Type',data=df)
plt.show()
df.replace({'Item_Fat_Content':{'low Fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace=True)
encoder=LabelEncoder()
df['Item_Identifier']=encoder.fit_transform(df['Item_Identifier'])
df['Item_Type']=encoder.fit_transform(df['Item_Type'])
df['Item_Fat_Content']=encoder.fit_transform(df['Item_Fat_Content'])
df['Outlet_Identifier']=encoder.fit_transform(df['Outlet_Identifier'])
df['Outlet_Size']=encoder.fit_transform(df['Outlet_Size'])
df['Outlet_Location_Type']=encoder.fit_transform(df['Outlet_Location_Type'])
df['Outlet_Type']=encoder.fit_transform(df['Outlet_Type'])
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
reg=xgb.XGBRegressor()
reg.fit(X_train,y_train)
y_train_pred=reg.predict(X_train)
y_test_pred=reg.predict(X_test)
acc1=r2_score(y_train,y_train_pred)
print("Training accuracy = ",acc1)
acc2=r2_score(y_test,y_test_pred)
print("Testing accuracy = ",acc2)

