#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers
get_ipython().system('pip install Pillow')
import PIL

get_ipython().system('pip install imageio')
import imageio


# In[2]:


##path to training data

images_dir = 'archive/seg_train/seg_train'
class_names = os.listdir(images_dir)
n_classes = len(class_names)


# In[3]:


class_names


# In[4]:


n_classes


# In[5]:


preprocess_gen = ImageDataGenerator(rescale=1./255.,
                                   #rotation_range=45,
                                   #width_shift_range=0.5,
                                   #height_shift_range=0.5,
                                   shear_range=5,
                                   #zoom_range=0.7,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                  )


# In[6]:


batch_size = 16


# In[7]:


train_generator_l = preprocess_gen.flow_from_directory(
    directory='archive/seg_train/seg_train',
    class_mode='categorical',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True,
    classes=class_names)

valid_generator_l = preprocess_gen.flow_from_directory(
    directory='archive/seg_test/seg_test',
    class_mode='categorical',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True,
    classes=class_names)  


# In[8]:


plt.hist(train_generator_l.classes, 
         bins=np.array([0, 1, 2, 3, 4, 5, 6])-0.5, rwidth=0.9)
plt.title('Class distribution');


# In[ ]:





# In[9]:


## some example plots
plt.figure(figsize=(20, 20))
for image, label in train_generator_l:
 

  
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.imshow(image[i])
    ax.set_title(class_names[np.argmax(label[i])])
    ax.axis("off")
  break


# In[ ]:





# In[10]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('archive/seg_train/seg_train/buildings/10006.jpg') #These two lines
im_arr = np.array(im) #are all you need
plt.imshow(im_arr) #Just to verify that image array has been constructed properly
im_arr.shape


# In[11]:


import glob


# In[12]:


#path to training images
trainingsbilder = glob.glob("archive/seg_train/seg_train/*/*.jpg")


# In[13]:


len(trainingsbilder)


# In[14]:


#keras imports

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import preprocess_input

from pickle import dump


# In[ ]:





# In[ ]:


#predict feature vector for any training image from the -2 layer of the vgg16
# done in batches of max 5000 pictures due to memory load


bilderstapel_1=[]
for k in trainingsbilder[0:5000]:  
    
   
    # load an image from file
    image = load_img(k, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # get extracted features
    features = model.predict(image)
    features = np.append(features, k)
    bilderstapel_1.append(features)
    print(k)
    print(len(bilderstapel_1))
    # save to file
    #dump(features, open('dog.pkl', 'wb'))
    print(features)

print(len(bilderstapel_1))
array_bilder_1 = np.array(bilderstapel_1)
import pandas as pd
df_1 = pd.DataFrame(array_bilder_1)
df_1.to_csv('df_1.csv')


# In[ ]:


bilderstapel_5000=[]
for k in trainingsbilder[5001:8000]:  
    
   
    # load an image from file
    image = load_img(k, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # get extracted features
    features = model.predict(image)
    features = np.append(features, k)
    bilderstapel_5000.append(features)
    print(k)
    print(len(bilderstapel_5000))
 
    print(features)
    
array_bilder_5000 = np.array(bilderstapel_5000)
import pandas as pd
df_5000 = pd.DataFrame(array_bilder_5000)
df_5000
df_5000.to_csv('df_5000.csv')


# In[ ]:


bilderstapel_8000=[]
for k in trainingsbilder[8001:11000]:  
    
   
    # load an image from file
    image = load_img(k, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # get extracted features
    features = model.predict(image)
    features = np.append(features, k)
    bilderstapel_8000.append(features)
    print(k)
    print(len(bilderstapel_8000))
    # save to file
    #dump(features, open('dog.pkl', 'wb'))
    print(features)
    
array_bilder_8000 = np.array(bilderstapel_8000)
import pandas as pd
df_8000 = pd.DataFrame(array_bilder_8000)
df_8000
df_8000.to_csv('df_8000.csv')


# In[ ]:


bilderstapel_11001=[]
for k in trainingsbilder[11001:14034]:  
    
   
    # load an image from file
    image = load_img(k, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # load model
    model = VGG16()
    # remove the output layer
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # get extracted features
    features = model.predict(image)
    features = np.append(features, k)
    bilderstapel_11001.append(features)
    print(k)
    print(len(bilderstapel_11001))
    # save to file
    #dump(features, open('dog.pkl', 'wb'))
    print(features)
    
array_bilder_11001 = np.array(bilderstapel_11001)
import pandas as pd
df_11001 = pd.DataFrame(array_bilder_11001)
df_11001
df_11001.to_csv('df_11001.csv')


# In[ ]:


# Concat all datasets to one Dataframe for further analysis

df_1 = pd.read_csv('df_1.csv')
df_2  = pd.read_csv('df_5000.csv')
df_3  = pd.read_csv('df_8000.csv')
df_4  = pd.read_csv('df_11001.csv')
df_test = pd.read_csv('df_test.csv')
liste = [df_1, df_2, df_3, df_4]
df_1 = df_1.append(liste, ignore_index = True)
df_total = df_1


# In[26]:


import pandas as pd
df_total = pd.read_csv('dataset_total.csv')


# In[27]:


df_total


# In[35]:


df_total.labls_num.unique()


# In[ ]:




