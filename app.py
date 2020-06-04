# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:28:14 2020

@author: divyam07
"""

from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Keras
from keras.models import load_model
from keras.preprocessing import image
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
# Flask utils
from flask import Flask,  request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Covid_19.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
#print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    preds = model.predict_classes(x)
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds= model_predict(file_path, model)
        if preds [0][0]== 0:
            pred = print ('Prediction : Please visit nearby covid centre , it seems you have symptoms')
        elif preds [0][0]== 1:
            pred =print('Prediction : Your X ray seems normal. But in case if you feel any symptoms please visit near by centre')        
        
        else:
            # Return empty response body and status code.
            pred = print ("Sorry Wrong Input")
        return pred
        
    return render_template('index.html', predict = pred )

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False,threaded=False)
