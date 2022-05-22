#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import random
import tensorflow as tf
import keras 


import random
import sys 

print(tf.__version__)
print(sys.version)


# In[2]:


import numpy as np
import matplotlib.pyplot as plt


# use seaborn plotting defaults
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageFilter


# In[ ]:





# In[3]:


loaded = tf.saved_model.load('./custom_best') 
converter = tf.lite.TFLiteConverter.from_saved_model("./custom_best")
tflite_best_custom = "./tflite_custom_best"
with open(tflite_best_custom, "wb") as f:
          f.write(converter.convert())


# In[4]:


with open(tflite_best_custom, 'rb') as f:
    tflite_best_custom_instance = f.read()
    


# In[5]:


interpreter_custom_best = tf.lite.Interpreter(model_content=tflite_best_custom_instance)
interpreter_custom_best.allocate_tensors()
input_index = interpreter_custom_best.get_input_details()[0]['index']
output_index = interpreter_custom_best.get_output_details()[0]['index']


# In[ ]:


normals = os.listdir('/bigdata3/OCT2017/all_data/NORMAL/')
drusen = os.listdir('/bigdata3/OCT2017/all_data/DRUSEN//')
dme = os.listdir('/bigdata3/OCT2017/all_data/DME/')
cnv = os.listdir('/bigdata3/OCT2017/all_data/CNV/')


# In[7]:


types = ['CNV', 'DME', 'DRUSEN', 'NORMAL'] 


# In[ ]:





# In[ ]:



                    


# In[25]:


normal_image_path = os.path.join('/bigdata3/OCT2017/all_data/NORMAL/', random.choice(normals))

print(f'testing image {normal_image_path}')
normal_image = tf.io.read_file(normal_image_path)
normal_image = tf.io.decode_jpeg(normal_image, channels=3)
fni = tf.image.resize(
    normal_image,
    (256, 256),
    # method=ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)
fni = np.float32(fni.numpy()/255.)
fni = fni.reshape((1, 256, 256, 3))
interpreter_custom_best.set_tensor(input_index, fni)
interpreter_custom_best.invoke()
result = interpreter_custom_best.get_tensor(output_index)
print(result)
amax = np.argmax(result)
print(amax)
percent = result[0][amax] * 100
image_type = types[amax]
print(f'type: {image_type} confidence: {percent}%')


# In[29]:


drusen_image_path = os.path.join('/bigdata3/OCT2017/all_data/DRUSEN/', random.choice(drusen))

print(f'testing image {drusen_image_path}')
drusen_image = tf.io.read_file(drusen_image_path)
drusen_image = tf.io.decode_jpeg(drusen_image, channels=3)
fni = tf.image.resize(
    drusen_image,
    (256, 256),
    # method=ResizeMethod.BILINEAR,
    preserve_aspect_ratio=False,
    antialias=False,
    name=None
)
fni = np.float32(fni.numpy()/255.)
fni = fni.reshape((1, 256, 256, 3))
interpreter_custom_best.set_tensor(input_index, fni)
interpreter_custom_best.invoke()
result = interpreter_custom_best.get_tensor(output_index)
print(result)
amax = np.argmax(result)
print(amax)
percent = result[0][amax] * 100
image_type = types[amax]
print(f'type: {image_type} confidence: {percent}%')


# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




