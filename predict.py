from __future__ import division, print_function
# coding=utf-8
import sys
import os
import re
import random


# Keras
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications import VGG16


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

import tensorflow as tf
from keras.models import load_model
MODEL_PATH = 'Tomoto_leaf_disease_prediction_final_model_VGG16y.h5'
model = load_model(MODEL_PATH)
model._make_predict_function()         
print('Model loaded. Check http://127.0.0.1:5000/')

img_width, img_height = 224, 224  # Default input size for VGG16
conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(img_width, img_height, 3))


from keras.preprocessing import image
global graph
graph = tf.get_default_graph()


def model_predict(img_path, model):
    org_img = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  
    img_tensor /= 255.  

    with graph.as_default():
        features = conv_base.predict(img_tensor.reshape(1,img_width, img_height, 3))
        
    

    prediction = model.predict(features)
    return prediction

    

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
        prediction = model_predict(file_path, model)
        classes = ["Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy", "Tomato Late Blight", "Tomato Leaf Mold","Tomato Mosaic Virus","Tomato Septoria Leaf Spot"]
        return str(classes[np.argmax(np.array(prediction[0]))])
    return None



if __name__ == '__main__':
    app.run(debug=True)

