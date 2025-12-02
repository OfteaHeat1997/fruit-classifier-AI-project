"""
Convert Keras model to TensorFlow Lite format.

TFLite = smaller, faster model for mobile apps and web deployment.

Usage:
    python scripts/convert_to_tflite.py
"""

import tensorflow as tf
import os

# Paths
MODEL_PATH = "models/fruit_classifier.keras"
TFLITE_PATH = "models/fruit_classifier.tflite"

def convert_to_tflite():
    """Convert Keras model to TensorFlow Lite format."""

    print("=" * 50)
    print("CONVERTING MODEL TO TENSORFLOW LITE")
    print("=" * 50)

    # Step 1: Load the Keras model
    print("\n[1/3] Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"      Model loaded from: {MODEL_PATH}")

    # Step 2: Convert to TFLite
    print("\n[2/3] Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimize for size (good for mobile)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    print("      Conversion complete!")

    # Step 3: Save the TFLite model
    print("\n[3/3] Saving TFLite model...")
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    # Show results
    keras_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

    print("\n" + "=" * 50)
    print("CONVERSION SUCCESSFUL!")
    print("=" * 50)
    print(f"\nOriginal Keras model:  {keras_size:.1f} MB")
    print(f"New TFLite model:      {tflite_size:.1f} MB")
    print(f"Size reduction:        {(1 - tflite_size/keras_size) * 100:.0f}%")
    print(f"\nSaved to: {TFLITE_PATH}")
    print("\nThis smaller model is ready for mobile apps!")

if __name__ == "__main__":
    convert_to_tflite()
