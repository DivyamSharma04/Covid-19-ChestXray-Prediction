# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:34:52 2020

@author: divyam07
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
model = load_model('model_covid.h5')
img = image.load_img('C://Users//divyam07//Desktop//COVID PREDICTION//Datasets//val//Covid//6CB4EFC6-68FA-4CD5-940C-BEFA8DAFE9A7.jpeg', target_size =(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis =0)
img_data = preprocess_input(x)
classes = model.predict(img_data)