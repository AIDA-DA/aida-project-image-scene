#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_1 = pd.read_csv('df_3924.csv')


# In[3]:


df_1


# In[4]:


df_glacier = df_1[df_1['4096'].str.contains("glacier")] 
df_glacier['label'] = 'glacier'  
df_glacier


# In[5]:


df_forest = df_1[df_1['4096'].str.contains("forest")] 
df_forest['label'] = 'forest'  
df_forest


# In[15]:




df_street = df_1[df_1['4096'].str.contains("street")] 
df_street['label'] = 'street'  
df_street


# In[16]:


df_buildings = df_1[df_1['4096'].str.contains("buildings")] 
df_buildings['label'] = 'buildings'  
df_buildings


# In[17]:


df_mountain = df_1[df_1['4096'].str.contains("mountain")] 
df_mountain['label'] = 'mountain'  
df_mountain


# In[18]:


df_sea = df_1[df_1['4096'].str.contains("sea")] 
df_sea['label'] = 'sea'  
df_sea


# In[ ]:




