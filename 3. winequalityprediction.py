# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:16:26 2023

@author: ds448
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv('winequality.csv')
print(df.shape)
print(df.head())
print(df.isnull().sum())
print(df.describe())
##plotting count acc to categories
sns.catplot(x='quality',data=df,kind='count')
plot=plt.figure(figsize=(5,5))
##Finding relation between quailty and volatile acidity
sns.barplot(x='quality',y='volatile acidity',data=df)
correlation=df.corr()
plot=plt.figure(figsize=(8,8))
##Checking correlation b/w features
sns.heatmap(correlation,cbar=True,square=True,annot=True)
X=df.iloc[:,:-1]
y=df['quality'].apply(lambda y_value:1 if y_value>=7 else 0)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_test=rf.predict(X_test)
accu_test=accuracy_score(y_test, y_pred_test)
print("Accuracy score: ",accu_test)
param={
       'n_estimators':[100,150,200,250,300],
       'criterion': ["gini", "entropy", "log_loss"], 
      'max_depth':[3,4,5,6]
       }
from sklearn.model_selection import GridSearchCV
gsv=GridSearchCV(rf, param_grid=param,cv=5,refit=True,verbose=3)
gsv.fit(X_train,y_train)
print(gsv.best_params_)
rf2=RandomForestClassifier(n_estimators=300,criterion='gini',max_depth=6)
rf2.fit(X_train,y_train)
y_pred_test2=rf.predict(X_test)
accu_test2=accuracy_score(y_test, y_pred_test2)
print("Accuracy score after cross validation: ",accu_test2)
input_data=(5.4,0.835,0.08,1.2,0.046,13.0,93.0,0.9924,3.57,0.85,13.0)
input_arr=np.asarray(input_data)
input_arr=input_arr.reshape(1,-1)
print("Predicted wine quality is: ",rf.predict(input_arr))
