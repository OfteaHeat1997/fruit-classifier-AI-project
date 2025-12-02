"""
Flask API for Fruit Ripeness Classifier

This is a simple web server that:
1. Receives an image from any frontend (HTML, React, mobile app)
2. Runs the AI prediction
3. Returns the result as JSON

Run with: python webapp/app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Allows React/mobile apps to connect
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
import os

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable cross-origin requests (needed for React/Expo)

# Load model and labels when server starts
print("Loading AI model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'fruit_classifier.tflite')
LABELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'class_labels.json')

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class labels (convert dict keys to list for easier indexing)
with open(LABELS_PATH, 'r') as f:
    class_labels_dict = json.load(f)
    # Convert {"0": "freshapples", "1": "freshbanana", ...} to ["freshapples", "freshbanana", ...]
    class_labels = [class_labels_dict[str(i)] for i in range(len(class_labels_dict))]

print(f"Model loaded! Classes: {class_labels}")


def preprocess_image(image_bytes):
    """
    Prepare image for the AI model.

    The model expects:
    - Size: 224x224 pixels
    - Format: RGB (3 color channels)
    - Values: 0 to 1 (normalized)
    """
    # Open image from bytes
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to RGB (in case it's PNG with alpha channel)
    img = img.convert('RGB')

    # Resize to 224x224 (what MobileNetV2 expects)
    img = img.resize((224, 224))

    # Convert to numpy array and normalize to 0-1
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Add batch dimension: (224,224,3) -> (1,224,224,3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(image_array):
    """
    Run AI prediction on preprocessed image.

    Returns: (predicted_class, confidence_percentage, all_probabilities)
    """
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get output (probabilities for each class)
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Find the class with highest probability
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index]) * 100

    # Get class name
    predicted_class = class_labels[predicted_index]

    # Create dictionary with all probabilities
    all_predictions = {
        class_labels[i]: float(predictions[i]) * 100
        for i in range(len(class_labels))
    }

    return predicted_class, confidence, all_predictions


# ============================================================
# ROUTES (URLs that the frontend can call)
# ============================================================

@app.route('/')
def home():
    """
    Home page - serves the HTML frontend.
    URL: http://localhost:5000/
    """
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions.
    URL: http://localhost:5000/api/predict

    How to use:
    - Send a POST request with an image file
    - Returns JSON with prediction results

    This is what React/Expo apps will call!
    """
    # Check if image was sent
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Get the image file
    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Preprocess for AI
        image_array = preprocess_image(image_bytes)

        # Get prediction
        predicted_class, confidence, all_predictions = predict(image_array)

        # Parse the class name (e.g., "freshapples" -> "Fresh Apples")
        # Format: "ripeness" + "fruit" -> "Ripeness Fruit"
        fruit_name = predicted_class.replace('fresh', 'Fresh ').replace('rotten', 'Rotten ').replace('unripe', 'Unripe ')
        fruit_name = fruit_name.replace('apples', 'Apple').replace('bananas', 'Banana').replace('oranges', 'Orange')

        # Return JSON result
        return jsonify({
            'success': True,
            'prediction': {
                'class': predicted_class,
                'label': fruit_name.strip(),
                'confidence': round(confidence, 1),
            },
            'all_predictions': all_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint - useful to verify server is running.
    URL: http://localhost:5000/api/health
    """
    return jsonify({
        'status': 'healthy',
        'model': 'fruit_classifier.tflite',
        'classes': class_labels
    })


# ============================================================
# RUN THE SERVER
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("FRUIT CLASSIFIER API SERVER")
    print("=" * 50)
    print("\nServer starting...")
    print("\nOpen in browser: http://localhost:5000")
    print("\nAPI endpoints:")
    print("  - GET  /           -> HTML frontend")
    print("  - POST /api/predict -> Send image, get prediction")
    print("  - GET  /api/health  -> Check server status")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50 + "\n")

    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=True)
