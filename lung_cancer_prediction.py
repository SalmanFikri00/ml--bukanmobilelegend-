# -*- coding: utf-8 -*-
"""Lung Cancer Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NBJuFVwiLsQjLTLPUSwWkl7sb8UncwOa
"""

# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)

train_folder = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/train'
test_folder = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/test'
validate_folder = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/valid'

normal_folder = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/train/normal'
adenocarcinoma_folder = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/train/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
large_cell_carcinoma_folder = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/train/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
squamous_cell_carcinoma_folder = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/train/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'

normal_folder_test = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/test/normal'
adenocarcinoma_folder_test = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/test/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
large_cell_carcinoma_folder_test = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/test/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
squamous_cell_carcinoma_folder_test = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/test/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'

normal_folder_validate = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/valid/normal'
adenocarcinoma_folder_validate = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/valid/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib'
large_cell_carcinoma_folder_validate = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/valid/large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa'
squamous_cell_carcinoma_folder_validate = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/valid/squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import utils

print("Libraries Imported")

# Define your folder paths
train_folder = '/home/slmnfkri/Desktop/project/train'
test_folder = '/home/slmnfkri/Desktop/project/test'
validate_folder = '/home/slmnfkri/Desktop/project/valid'

# Define the image size
IMAGE_SIZE = (350, 350)

print("Reading training images from:", train_folder)
print("Reading validation images from:", validate_folder)

# Set up data augmentation for training and test data
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 8

# Create the train generator for loading images
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

# Create the validation generator for loading images
validation_generator = test_datagen.flow_from_directory(
    validate_folder,  # Use validate_folder here for validation data
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode='categorical'
)

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Reduce learning rate when the loss stops improving
learning_rate_reduction = ReduceLROnPlateau(
    monitor='loss',       # Monitors the loss
    patience=5,           # Waits for 5 epochs with no improvement
    verbose=2,            # Prints a message when the learning rate is reduced
    factor=0.5,           # Reduces the learning rate by a factor of 0.5
    min_lr=0.000001       # Sets a lower bound for the learning rate
)

# Stop training if the loss does not improve
early_stops = EarlyStopping(
    monitor='loss',       # Monitors the loss
    min_delta=0,          # Minimum change to qualify as an improvement
    patience=6,           # Stops after 6 epochs with no improvement
    verbose=2,            # Prints a message when stopping
    mode='auto'           # Automatically decides whether to minimize or maximize the loss
)

# Save the model (architecture + weights) when the model achieves the best performance
checkpointer = ModelCheckpoint(filepath='/content/best_model.keras', verbose=2, save_best_only=True, save_weights_only=False)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import Xception

# Set image size
IMAGE_SIZE = (350, 350)

# Pretrained model (Xception) without the top layer
pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=[*IMAGE_SIZE, 3])

# Freeze the pretrained layers
pretrained_model.trainable = False

# Create the final model
model = Sequential()
model.add(pretrained_model)
model.add(GlobalAveragePooling2D())  # Add GlobalAveragePooling
model.add(Dense(4, activation='softmax'))  # Output layer with 4 classes

# Build the model to define the input shape
model.build(input_shape=(None, 350, 350, 3))

# Show the summary of the pretrained model
print("Pretrained model used:")
pretrained_model.summary()

# Show the summary of the final model
print("Final model created:")
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=25,
    epochs=50,
    callbacks=[learning_rate_reduction, early_stops, checkpointer],
    validation_data=validation_generator,
    validation_steps=20
)

print("Final training accuracy =", history.history['accuracy'][-1])
print("Final testing accuracy =", history.history['val_accuracy'][-1])

def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)

model.save('/content/drive/MyDrive/dataset/trained_lung_cancer_model.hdf5')

from tensorflow.keras.preprocessing import image
import numpy as np

# Define a function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Load an image from your drive
img_path = '/content/drive/MyDrive/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning/dataset/valid/adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/000112 (9).png'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)

# Make a prediction
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])

# Map the predicted class to the class label
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[predicted_class]

# Print the predicted class
print(f"The image belongs to class: {predicted_label}")

# Display the image
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np

# Define a function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Load an image from your drive
img_path = '/content/thumb.jpg'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)

# Make a prediction
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])

# Map the predicted class to the class label
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[predicted_class]

# Print the predicted class
print(f"The image belongs to class: {predicted_label}")

# Display the image
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np

# Define a function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Load an image from your drive
img_path = '/content/l3.png'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)

# Make a prediction
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])

# Map the predicted class to the class label
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[predicted_class]

# Print the predicted class
print(f"The image belongs to class: {predicted_label}")

# Display the image
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

from tensorflow.keras.preprocessing import image
import numpy as np

# Define a function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image like the training images
    return img_array

# Load an image from your drive
img_path = '/content/n8.jpg'
img = load_and_preprocess_image(img_path, IMAGE_SIZE)

# Make a prediction
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])

# Map the predicted class to the class label
class_labels = list(train_generator.class_indices.keys())
predicted_label = class_labels[predicted_class]

# Print the predicted class
print(f"The image belongs to class: {predicted_label}")

# Display the image
plt.imshow(image.load_img(img_path, target_size=IMAGE_SIZE))
plt.title(f"Predicted: {predicted_label}")
plt.axis('off')
plt.show()

