#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd


# In[46]:


df_1 = pd.read_csv('df_3924.csv')
df_2 = pd.read_csv('df_3925.csv')
df_3  = pd.read_csv('df_5000.csv')
df_4  = pd.read_csv('df_8000.csv')
df_5  = pd.read_csv('df_11001.csv')
df_test = pd.read_csv('df_test.csv')
liste = [df_2, df_3, df_4, df_test, df_5]


# In[47]:


df_1 = df_1.append(liste, ignore_index = True)
df_1


# In[48]:


class_names = ['glacier', 'forest', 'street', 'buildings', 'mountain', 'sea']
df_1['labls'] = df_1['4096']  
for k in class_names: 

    df_1['labls'] = df_1['labls'].apply(lambda x: k if k in x else x)

df_1


# In[49]:


df_1['labls'].nunique()


# In[50]:


df_1["labls"] = df_1["labls"].astype('category')
df_1.dtypes
df_1["labls_num"] = df_1["labls"].cat.codes
df_1.head()
df_1.to_csv('dataset_total.csv')


# In[51]:


df_1


# In[52]:


df_1.to_numpy()


# In[18]:


import numpy as np
import sys
get_ipython().system('{sys.executable} -m pip install sklearn')

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import StandardScaler
X = df_1.iloc[:,0:4095].values
y = df_1["labls"]


# commonly done in classification: stratify by target variable y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape)
print(X_test.shape)

from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel

clf.fit(X_train, y_train)
#Create a svm Classifier


#Train the model using the training sets

a=clf.score(X_test, y_test)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
print(a)


# In[44]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X = df_1.iloc[:,0:4095].values
y = df_1["labls"]

clf = LogisticRegression(random_state=0, max_iter = 2000).fit(X_train, y_train)

clf.predict(X_test)

clf.predict_proba(X_test)


clf.score(X_test, y_test)


# In[53]:


X = df_1.iloc[:,0:4095].values
y = df_1["labls"]


# In[54]:


from sklearn.cluster import KMeans
import numpy as np
X = df_1.iloc[:,0:4095].values
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
kmeans.labels_

df_1['cluster']=kmeans.predict(X)

kmeans.cluster_centers_


# In[21]:


df_1


# In[56]:


pd.crosstab(df_1.cluster, df_1.labls,margins=True)


# In[ ]:




