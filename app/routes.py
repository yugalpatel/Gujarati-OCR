from flask import Blueprint, render_template, request, redirect, url_for, current_app
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

bp = Blueprint('routes', __name__)

# Model download logic
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # Replace with your actual file ID
model_path = 'models/efficientnetb3_gujarati.h5'

def download_model_if_needed():
    if not os.path.exists(model_path):
        print("Model not found, downloading...")
        gdown.download(MODEL_URL, model_path, quiet=False)
    else:
        print("Model already exists.")

download_model_if_needed()

# Load model
def custom_objects():
    from tensorflow.keras.layers import DepthwiseConv2D
    return {'DepthwiseConv2D': DepthwiseConv2D}

model = load_model(model_path, custom_objects=custom_objects())
class_df = pd.read_csv('models/class_dict.csv')
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
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = Image.open(filepath)
            img = preprocess_image(img, (40, 40))
            predictions = model.predict(img)
            predicted_class = class_indices[np.argmax(predictions)]

            return render_template('result.html', filename=file.filename, prediction=predicted_class)

    return render_template('index.html')


