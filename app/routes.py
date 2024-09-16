from flask import Blueprint, render_template, request, redirect, url_for
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.layers import DepthwiseConv2D

bp = Blueprint('routes', __name__)

# Load the model and class dictionary
model_path = 'models/efficientnetb3_gujarati.h5'
class_dict_path = 'models/class_dict.csv'

def custom_objects():
    class CustomDepthwiseConv2D(DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            if 'groups' in kwargs:
                del kwargs['groups']
            super().__init__(*args, **kwargs)

    return {'DepthwiseConv2D': CustomDepthwiseConv2D}

model = load_model(model_path, custom_objects=custom_objects())
class_df = pd.read_csv(class_dict_path)
class_indices = class_df.set_index('class_index')['class'].to_dict()

def preprocess_image(image, img_size):
    image = image.resize(img_size)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)

            # Process the image and predict
            img_size = (40, 40)
            img = Image.open(filepath)
            img = preprocess_image(img, img_size)
            predictions = model.predict(img)
            predicted_class = class_indices[np.argmax(predictions)]

            return render_template('result.html', filename=filename, prediction=predicted_class)
    
    return render_template('index.html')

@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)



