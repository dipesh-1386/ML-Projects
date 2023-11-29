# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:05:02 2023

@author: ds448
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
df=pd.read_csv('Mall_Customers.csv')
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
X=df.iloc[:,[3,4]].values
## Find wcss-> within clusters sum of squares
##By using elbow method
wcss=[]
for i in range(1,14):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X)
    ##It .inertia gives wcss value
    wcss.append(kmeans.inertia_)
##Plotting an elbow graph
plt.plot(range(1,14),wcss)
plt.title('Elbow graph')
plt.xlabel('Values')
plt.ylabel('WCSS value')
plt.show()
##Optimum value of K=5
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=10)
kmeans.fit(X)
## return a label for each datapoint(this label means the number given to each cluster) and this will be taken as output
y=kmeans.fit_predict(X)
##Visualizing the clusters by plotting the clusters and centroids
plt.figure(figsize=(12,6))
plt.scatter(X[y==0,0],X[y==0,1],s=50,c='red',label='Cluster 1')
plt.scatter(X[y==1,0],X[y==1,1],s=50,c='blue',label='Cluster 2')
plt.scatter(X[y==2,0],X[y==2,1],s=50,c='orange',label='Cluster 3')
plt.scatter(X[y==3,0],X[y==3,1],s=50,c='green',label='Cluster 4')
plt.scatter(X[y==4,0],X[y==4,1],s=50,c='brown',label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='black',s=100,label='Centroids')
plt.title('Customer Groups by k-means')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()
