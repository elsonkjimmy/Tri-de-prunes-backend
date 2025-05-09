from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from google.generativeai import configure, GenerativeModel

import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io

# Load environment
load_dotenv()

# Initialize app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Load Gemini config
configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load model
model = tf.keras.models.load_model("plum_classifier_model.h5")

# Classes
class_names = ["Anjou", "Black Splendor", "Friar", "Green Gage", "President", "Stanley"]

# Resize + preprocessing
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file is None:
        return jsonify({'error': 'Aucune image reçue'}), 400

    try:
        image_bytes = file.read()
        input_image = preprocess_image(image_bytes)

        predictions = model.predict(input_image)[0]
        label = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # GradCAM simplifié : ici juste renvoyer l'image originale encodée
        gradcam_encoded = base64.b64encode(image_bytes).decode('utf-8')
        gradcam_image = f"data:image/jpeg;base64,{gradcam_encoded}"

        return jsonify({
            'label': label,
            'confidence': confidence,
            'gradcamImage': gradcam_image
        })

    except Exception as e:
        print("❌ Erreur de prédiction:", e)
        return jsonify({'error': 'Erreur lors de la prédiction'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt')

    try:
        print("🟡 Prompt envoyé à Gemini:", prompt)
        model = GenerativeModel('models/gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return jsonify({ 'text': response.text })

    except Exception as e:
        print("❌ Erreur Gemini:", e)
        return jsonify({ 'error': 'Erreur lors de la génération de contenu' }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
