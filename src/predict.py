#!/usr/bin/env python3
"""
Flask web service for flower classification predictions.

Endpoints:
    POST /predict - Classify a flower image
    GET /health  - Health check

Usage:
    python predict.py
"""

import os
import io
import base64
import numpy as np
import requests
import tensorflow as tf
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image

# Configuration
BASE_DIR = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "flower_classifier.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.txt"
IMG_SIZE = (224, 224)
PORT = int(os.environ.get("PORT", 9696))

# Initialize Flask app
app = Flask(__name__)

# Global model (loaded once at startup)
model = None
class_names = None


def load_model():
    """Load the trained model and class names."""
    global model, class_names
    
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded")
    
    print(f"Loading class names from {CLASSES_PATH}...")
    with open(CLASSES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"✓ Classes: {class_names}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess an image for prediction."""
    
    # Resize
    image = image.resize(IMG_SIZE)
    
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_image(image: Image.Image) -> dict:
    """Run prediction on an image."""
    
    # Preprocess
    img_array = preprocess_image(image)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    
    # Get top prediction
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_idx]
    confidence = float(probabilities[predicted_idx])
    
    # Build probability dict
    prob_dict = {
        class_names[i]: float(probabilities[i])
        for i in range(len(class_names))
    }
    
    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in prob_dict.items()}
    }


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict flower type from an image.
    
    Accepts:
        - JSON with "image_url": URL to fetch image from
        - JSON with "image_base64": Base64-encoded image data
        - Multipart form with "image": Image file upload
    """
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        image = None
        
        # Check for file upload
        if "image" in request.files:
            file = request.files["image"]
            image = Image.open(file.stream)
        
        # Check for JSON body
        elif request.is_json:
            data = request.get_json()
            
            # Option 1: Image URL
            if "image_url" in data:
                url = data["image_url"]
                headers = {
                    "User-Agent": "FlowerClassifier/1.0 (ML Zoomcamp Project; https://github.com/HighviewOne/flower-classification-capstone)"
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
            
            # Option 2: Base64-encoded image
            elif "image_base64" in data:
                img_data = base64.b64decode(data["image_base64"])
                image = Image.open(io.BytesIO(img_data))
            
            else:
                return jsonify({
                    "error": "No image provided. Use 'image_url', 'image_base64', or file upload."
                }), 400
        
        else:
            return jsonify({
                "error": "No image provided. Use 'image_url', 'image_base64', or file upload."
            }), 400
        
        # Run prediction
        result = predict_image(image)
        return jsonify(result)
    
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 400
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


def main():
    """Start the prediction service."""
    
    print("=" * 50)
    print("FLOWER CLASSIFICATION - PREDICTION SERVICE")
    print("=" * 50)
    
    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run 'python src/train.py' first.")
        return
    
    # Load model
    load_model()
    
    # Start server
    print(f"\nStarting server on port {PORT}...")
    print(f"Endpoints:")
    print(f"  POST http://localhost:{PORT}/predict")
    print(f"  GET  http://localhost:{PORT}/health")
    print("-" * 50)
    
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
