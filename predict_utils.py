# predict_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Charger le modèle déjà entraîné
model = tf.keras.models.load_model("plum_classifier_model.h5")

# Charger les classes
class_labels = ['bruised', 'cracked', 'rotten', 'spotted','unaffected', 'unripe']  
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = class_labels[predicted_index]

    return predicted_label
