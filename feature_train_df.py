#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


#load testsets

df_1 = pd.read_csv('df_3924.csv')
df_2 = pd.read_csv('df_3925.csv')
df_3  = pd.read_csv('df_5000.csv')
df_4  = pd.read_csv('df_8000.csv')
df_5  = pd.read_csv('df_11001.csv')
df_test = pd.read_csv('df_test.csv')
liste = [df_2, df_3, df_4, df_test, df_5]


# In[10]:


df_1 = df_1.append(liste, ignore_index = True)
df_1


# In[11]:


class_names = ['glacier', 'forest', 'street', 'buildings', 'mountain', 'sea']
df_1['labls'] = df_1['4096']  
for k in class_names: 

    df_1['labls'] = df_1['labls'].apply(lambda x: k if k in x else x)

df_1


# In[12]:


df_1['labls'].nunique()


# In[13]:


df_1["labls"] = df_1["labls"].astype('category')
df_1.dtypes
df_1["labls_num"] = df_1["labls"].cat.codes
df_1.head()
df_1.to_csv('dataset_total.csv')


# In[14]:


df_1


# In[9]:


df_1.to_numpy()


# In[15]:


import numpy as np
import sys
import random
get_ipython().system('{sys.executable} -m pip install sklearn')

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X = df_1.iloc[:,0:4095].values
y = df_1["labls"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



from sklearn import svm

clf = svm.SVC(kernel='linear') # Linear Kernel
#clf = LogisticRegression(random_state=0, max_iter = 2000).fit(X_train, y_train)

#Train the model using the training sets
clf.fit(X_train, y_train)


#print accuracy score for test set
a=clf.score(X_test, y_test)
print(a, r, )

#Predict the response for whole dataset
y_pred = clf.predict(X_test)

# write predictions to a new dataframe column
df_1['predicit_class'] = ypred


# In[ ]:


#same procedure as above for alternative Classifier (logistic Regression)

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X = df_1.iloc[:,0:4095].values
y = df_1["labls"]

clf = LogisticRegression(random_state=0, max_iter = 2000).fit(X_train, y_train)

clf.predict(X_test)

clf.predict_proba(X_test)


clf.score(X_test, y_test)

