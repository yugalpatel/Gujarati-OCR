from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

bp = Blueprint('routes', __name__)

# URL to download the model from Google Drive
MODEL_URL = "https://drive.google.com/file/d/1x9hSSGzEnLDhMPzKTDzJ8O-Zd1HcepqX/view?usp=sharing"  # replace YOUR_FILE_ID with your Google Drive file ID

def download_model_if_needed():
    model_path = 'models/efficientnetb3_gujarati.h5'
    if not os.path.exists(model_path):
        print("Model not found, downloading...")
        gdown.download(MODEL_URL, model_path, quiet=False)
    else:
        print("Model already exists.")

# Call this function before loading the model
download_model_if_needed()

# Load the model and class dictionary
model_path = 'models/efficientnetb3_gujarati.h5'
class_dict_path = 'models/class_dict.csv'

def custom_objects():
    from tensorflow.keras.layers import DepthwiseConv2D
    custom_objs = {'DepthwiseConv2D': DepthwiseConv2D}
    return custom_objs

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

    # Debug print statement to check template directory
    print("Template folder:", current_app.template_folder)
    print("Looking for template: index.html")
    return render_template('index.html')

@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

