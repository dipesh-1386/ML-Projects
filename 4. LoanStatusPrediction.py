# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:45:27 2023

@author: ds448
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
df=pd.read_csv("loanstatus.csv")
print(df.shape)
print(df.head())
print(df.describe())
print(df.isnull().sum())
##dropping mssing values
df=df.dropna()
print(df.isnull().sum())
print("Count of total categories in property area = ",df['Property_Area'].value_counts())
df.replace({'Loan_Status':{'N':0,'Y':1},"Dependents":{'3+':4},'Gender':{'Male':1,'Female':0}},inplace=True)
df.replace({'Education':{'Graduate':1,'Not Graduate':0},'Married':{'Yes':1,'No':0},'Self_Employed':{'Yes':1,'No':0}},inplace=True)
df.replace({'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2}},inplace=True)
sns.countplot(x='Education',hue='Loan_Status',data=df)
sns.countplot(x='Married',hue='Loan_Status',data=df)
df['Dependents']=df['Dependents'].astype(int)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
X=X.drop(columns='Loan_ID',axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
classifier=svm.SVC(kernel='linear',verbose=3)
classifier.fit(X_train,y_train)
y_pred_train=classifier.predict(X_train)
y_pred_test=classifier.predict(X_test)
train_acc=accuracy_score(y_train,y_pred_train)
print("Training accuracy: ",train_acc)
test_acc=accuracy_score(y_test,y_pred_test)
print("Test accuracy: ",test_acc)