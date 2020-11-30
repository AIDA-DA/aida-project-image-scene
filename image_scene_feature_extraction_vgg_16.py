#!/usr/bin/env python
# coding: utf-8

# In[1]:


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



plt.figure(figsize=(20, 20))
for image, label in train_generator_l:
 

  
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.imshow(image[i])
    ax.set_title(class_names[np.argmax(label[i])])
    ax.axis("off")
  break


# In[ ]:


image


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


trainingsbilder = glob.glob("archive/seg_train/seg_train/*/*.jpg")


# In[13]:


len(trainingsbilder)


# In[16]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump


# In[ ]:


bilderstapel_3925=[]
for k in trainingsbilder[3925:14034]:  
    
   
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
    bilderstapel_3925.append(features)
    print(k)
    print(len(bilderstapel_3925))
    # save to file
    #dump(features, open('dog.pkl', 'wb'))
    print(features)


# In[47]:


print(len(bilderstapel))
array_bilder_3925 = np.array(bilderstapel_3925)


# In[48]:


import pandas as pd
df_3925 = pd.DataFrame(array_bilder_3925)


# In[50]:



df_3925.to_csv('df_3925.csv')


# In[46]:


df_glacier = df[df[4096].str.contains("glacier")] 
df_glacier['label'] = 'glacier'  
df_glacier


# In[ ]:




