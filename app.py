from flask import Flask, request, jsonify
import tensorflow as tf
from flask_cors import CORS
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

#CORS 
CORS(app)

# Load model yang udah kamu train
model = tf.keras.models.load_model('./trained_lung_cancer_model.hdf5')

# Image size (sesuai ukuran yang dipakai di model kamu)
IMAGE_SIZE = (350, 350)

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")  # Convert to RGB to ensure 3 channels
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    img_file = request.files['file']
    img_path = f"/tmp/{img_file.filename}"
    img_file.save(img_path)

    # Preprocess and predict
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    # Convert class index to label
    class_labels = ['normal', 'adenocarcinoma', 'large_cell_carcinoma', 'squamous_cell_carcinoma']
    predicted_label = class_labels[predicted_class]

    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
