"""
Fruit Ripeness Classifier - Prediction Script
Student: Maria Paula Salazar Agudelo
Course: AI Minor - Personal Challenge

WHAT THIS DOES:
1. Loads the trained model
2. Takes an image file path as input
3. Preprocesses the image (resize, normalize)
4. Makes a prediction
5. Saves prediction to database
6. Shows the result with confidence

HOW TO RUN:
    python scripts/predict.py path/to/image.jpg

    Example:
    python scripts/predict.py test_images/apple1.jpg

REQUIREMENTS:
- Trained model in: models/fruit_classifier.keras
- Class labels in: models/class_labels.json
- Python packages: tensorflow, numpy, pillow

OUTPUT:
- Prints prediction with confidence
- Saves to predictions.db database
- Returns prediction result
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Import database helper functions
from db_helper import init_database, log_prediction

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model files paths
MODEL_PATH = "models/fruit_classifier.keras"
LABELS_PATH = "models/class_labels.json"

# Image preprocessing parameters (must match training)
IMG_SIZE = 224  # MobileNetV2 requirement

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model_and_labels():
    """
    Load the trained model and class labels.

    WHAT IT DOES:
    1. Loads the saved Keras model file
    2. Loads the JSON file with class names

    RETURNS:
        model: Trained Keras model ready to make predictions
        class_labels: Dictionary mapping numbers to fruit names
                     Example: {0: "freshapples", 1: "freshbanana", ...}

    ERROR HANDLING:
    - Checks if files exist before loading
    - Shows helpful error message if files missing

    BEGINNER NOTE:
        The model file contains all the neural network weights
        The labels file tells what each number means
        (Model outputs numbers 0-8, which get converted to fruit names)
    """

    print("Loading model and class labels...")

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found!")
        print(f"Looking for: {MODEL_PATH}")
        print()
        print("Please train the model first:")
        print("  python scripts/train.py")
        sys.exit(1)

    # Check if labels file exists
    if not os.path.exists(LABELS_PATH):
        print(f"ERROR: Class labels file not found!")
        print(f"Looking for: {LABELS_PATH}")
        print()
        print("Please train the model first (creates this file):")
        print("  python scripts/train.py")
        sys.exit(1)

    # Load the model
    # WHY? This loads all the neural network weights from training
    model = keras.models.load_model(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    # Load class labels
    # WHY? Need to convert model output (numbers) to fruit names
    with open(LABELS_PATH, 'r') as f:
        class_labels = json.load(f)

    # Convert keys to integers
    # WHY? JSON stores keys as strings, but integers are needed for indexing
    # Example: {"0": "freshapples"} becomes {0: "freshapples"}
    class_labels = {int(k): v for k, v in class_labels.items()}

    print(f"Class labels loaded: {len(class_labels)} classes")

    return model, class_labels


def load_and_preprocess_image(image_path):
    """
    Load and preprocess an image for model prediction.

    WHAT IT DOES:
    1. Opens the image file
    2. Resizes to 224x224 (model requirement)
    3. Converts to array
    4. Normalizes pixel values (0-255 to 0-1)
    5. Adds batch dimension

    PARAMETERS:
        image_path (str): Path to image file

    RETURNS:
        img_array: Preprocessed image ready for prediction
                  Shape: (1, 224, 224, 3)
                  Values: 0.0 to 1.0

    WHY THESE STEPS?
        Resize: Model was trained on 224x224 images
        Normalize: Model expects values 0-1, not 0-255
        Batch dimension: Model expects batch of images, even if just 1

    BEGINNER NOTE:
        Think of preprocessing as "formatting" the image
        so the model can understand it
    """

    # Step 1: Open image
    # WHY? Load the image file into memory
    img = Image.open(image_path)

    # Step 2: Convert to RGB
    # WHY? Some images might be grayscale or RGBA
    # Model expects RGB (3 color channels)
    img = img.convert('RGB')

    # Step 3: Resize to 224x224
    # WHY? Model was trained on this size, must use same size
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Step 4: Convert to NumPy array
    # WHY? Model works with arrays, not PIL images
    # Shape: (224, 224, 3) - height, width, color channels
    img_array = np.array(img)

    # Step 5: Normalize pixel values
    # WHY? Convert from 0-255 to 0-1 (what model expects)
    # Example: pixel value 255 becomes 1.0, value 127 becomes 0.498
    img_array = img_array.astype('float32') / 255.0

    # Step 6: Add batch dimension
    # WHY? Model expects shape (batch_size, height, width, channels)
    # We have: (224, 224, 3)
    # We need: (1, 224, 224, 3)
    # The 1 means "batch of 1 image"
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(model, class_labels, image_path, save_to_db=True, source='script'):
    """
    Make a prediction on an image and optionally save to database.

    WHAT IT DOES:
    1. Loads and preprocesses the image
    2. Feeds image through the model
    3. Gets prediction and confidence
    4. Optionally saves to database
    5. Returns prediction results

    PARAMETERS:
        model: Trained Keras model
        class_labels: Dictionary of class names
        image_path: Path to image file
        save_to_db: Whether to save to database (default: True)
        source: Where prediction came from (default: 'script')

    RETURNS:
        Dictionary with:
            'predicted_label': Fruit name (e.g., "freshapples")
            'confidence': Confidence percentage 0-100
            'all_probabilities': Probabilities for all 9 classes

    HOW PREDICTION WORKS:
        1. Model outputs 9 probabilities (one per class)
        2. Example: [0.05, 0.02, 0.85, 0.01, 0.03, 0.01, 0.02, 0.01, 0.00]
        3. Find the highest probability (0.85 at index 2)
        4. Look up what class 2 means (e.g., "freshapples")
        5. Confidence = 85%
    """

    print()
    print("="*70)
    print("MAKING PREDICTION")
    print("="*70)
    print(f"Image: {image_path}")
    print()

    # Step 1: Check if image exists
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found!")
        print(f"Looking for: {image_path}")
        sys.exit(1)

    # Step 2: Load and preprocess image
    print("Preprocessing image...")
    img_array = load_and_preprocess_image(image_path)
    print(f"  Shape: {img_array.shape}")
    print(f"  Value range: {img_array.min():.3f} to {img_array.max():.3f}")

    # Step 3: Make prediction
    # WHY? This runs the image through all model layers
    # Input: (1, 224, 224, 3) preprocessed image
    # Output: (1, 9) probabilities for each class
    print()
    print("Running through neural network...")
    predictions = model.predict(img_array, verbose=0)
    # verbose=0 means don't show progress bar (cleaner output)

    # Step 4: Get predicted class
    # WHY? Find which class has highest probability
    # np.argmax returns the index of the maximum value
    # Example: [0.05, 0.85, 0.10] -> argmax = 1
    predicted_class_idx = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class_idx]

    # Step 5: Get confidence
    # WHY? How confident is the model in this prediction?
    # predictions[0] = array of 9 probabilities
    # predictions[0][predicted_class_idx] = probability of predicted class
    # Multiply by 100 to get percentage
    confidence = predictions[0][predicted_class_idx] * 100

    # Step 6: Display results
    print()
    print("PREDICTION RESULT:")
    print(f"  Predicted: {predicted_label}")
    print(f"  Confidence: {confidence:.2f}%")
    print()

    # Step 7: Show top 3 predictions
    # WHY? Sometimes useful to see what else the model considered
    print("Top 3 predictions:")
    # Get indices of top 3 probabilities (highest to lowest)
    top_3_indices = np.argsort(predictions[0])[::-1][:3]
    for i, idx in enumerate(top_3_indices, 1):
        label = class_labels[idx]
        prob = predictions[0][idx] * 100
        print(f"  {i}. {label}: {prob:.2f}%")

    # Step 8: Save to database
    if save_to_db:
        print()
        print("Saving to database...")
        log_prediction(
            image_path=image_path,
            predicted_label=predicted_label,
            confidence=confidence,
            source=source
        )

    # Step 9: Return results
    return {
        'predicted_label': predicted_label,
        'confidence': confidence,
        'all_probabilities': predictions[0].tolist()  # Convert to list for JSON
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function for prediction script.

    COMMAND LINE USAGE:
        python scripts/predict.py <image_path>

    EXAMPLE:
        python scripts/predict.py test_images/apple1.jpg
    """

    print("="*70)
    print("FRUIT RIPENESS CLASSIFIER - PREDICTION")
    print("="*70)
    print("Student: Maria Paula Salazar Agudelo")
    print("Course: AI Minor - Personal Challenge")
    print()

    # Step 1: Check command line arguments
    # WHY? User must provide image path
    if len(sys.argv) < 2:
        print("ERROR: No image path provided!")
        print()
        print("USAGE:")
        print("  python scripts/predict.py <image_path>")
        print()
        print("EXAMPLE:")
        print("  python scripts/predict.py test_images/apple1.jpg")
        sys.exit(1)

    # Step 2: Get image path from command line
    # sys.argv[0] = script name
    # sys.argv[1] = first argument (image path)
    image_path = sys.argv[1]

    # Step 3: Initialize database (creates table if doesn't exist)
    # WHY? Make sure database is ready before trying to save prediction
    init_database()

    # Step 4: Load model and labels
    model, class_labels = load_model_and_labels()

    # Step 5: Make prediction
    result = predict_image(
        model=model,
        class_labels=class_labels,
        image_path=image_path,
        save_to_db=True,
        source='script'
    )

    # Step 6: Summary
    print()
    print("="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print(f"Result: {result['predicted_label']} ({result['confidence']:.1f}% confidence)")
    print("Saved to database: predictions.db")
    print()


# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    This block runs when you execute this script directly.

    BEGINNER NOTE:
        if __name__ == "__main__" means:
        "Only run this if the script is executed directly,
         not if it's imported as a module"
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrediction interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        print()
        print("Troubleshooting:")
        print("  1. Check that model exists: models/fruit_classifier.keras")
        print("  2. Check that labels exist: models/class_labels.json")
        print("  3. Check that image path is correct")
        print("  4. Make sure TensorFlow is installed: pip install tensorflow")
        raise
