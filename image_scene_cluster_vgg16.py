#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# script to cluster images based on 4096 vgg16 features to check the labeling accuracy of original labels


# In[1]:


# imports
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


# In[2]:


# do the clustering
df_1 = pd.read_csv('dataset_total.csv')
X = df_1.iloc[:,0:4095].values
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
kmeans.labels_

df_1['cluster']=kmeans.predict(X)

kmeans.cluster_centers_


# In[3]:


# crosstabulate original and new labels

pd.crosstab(df_1.cluster, df_1.labls,margins=True)


# In[ ]:




