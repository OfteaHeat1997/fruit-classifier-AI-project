"""
Fruit Ripeness Classifier - Training Script
Student: Maria Paula Salazar Agudelo
Course: AI Minor - Personal Challenge

This script trains a deep learning model to classify fruit ripeness.

WHAT THIS DOES:
1. Loads fruit images from dataset folders
2. Builds a neural network using transfer learning
3. Trains the model for 20 epochs
4. Saves the trained model for predictions

HOW TO RUN:
    python scripts/train.py

REQUIREMENTS:
- Dataset in: data/fruit_ripeness_dataset/.../dataset/
- Python packages: tensorflow, numpy, pillow
- Recommended: GPU for faster training (2-3 min/epoch vs 15-20 min/epoch)

OUTPUT:
- models/fruit_classifier.keras (trained model)
- models/class_labels.json (class names)
- models/training_config.json (training info)
- models/training_history.png (accuracy/loss graphs)
"""

import os
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================================
# CONFIGURATION - Adjust these if needed
# ============================================================================

# Dataset paths (adjust if your data is elsewhere)
DATA_ROOT = r"C:\Users\maria\Desktop\fruit_ripeness\data\fruit_ripeness_dataset\fruit_ripeness_dataset\fruit_archive\dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# Training parameters
IMG_SIZE = 224          # Image size (MobileNetV2 requirement)
BATCH_SIZE = 32         # Images processed at once
EPOCHS = 20             # Training rounds
LEARNING_RATE = 0.0001  # Learning speed (slow = careful learning)

# Output directory
OUTPUT_DIR = "models"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a section header for better readability"""
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")

def create_data_generators():
    """
    Create data generators for training and testing.

    Data generators:
    - Load images from folders
    - Resize to 224x224
    - Normalize pixel values (0-255 to 0-1)
    - Apply augmentation to training images

    Returns:
        train_generator: Training data with augmentation
        test_generator: Test data without augmentation
    """
    print("Setting up data generators...")

    # Training data augmentation
    # WHY? Creates variations of images so model learns better
    train_datagen = ImageDataGenerator(
        rescale=1./255,              # Normalize: 0-255 to 0-1
        rotation_range=20,           # Rotate randomly ±20 degrees
        width_shift_range=0.2,       # Shift horizontally ±20%
        height_shift_range=0.2,      # Shift vertically ±20%
        zoom_range=0.2,              # Zoom in/out ±20%
        horizontal_flip=True,        # Flip horizontally randomly
        fill_mode='nearest'          # Fill empty pixels
    )

    # Test data: only normalize (no augmentation)
    # WHY? We want to test on original images
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Load training images
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),  # Resize to 224x224
        batch_size=BATCH_SIZE,
        class_mode='categorical',           # Multi-class classification
        shuffle=True                        # Shuffle for better learning
    )

    # Load test images
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False                       # Don't shuffle test data
    )

    print(f"Training images: {train_generator.samples}")
    print(f"Test images: {test_generator.samples}")
    print(f"Classes found: {len(train_generator.class_indices)}")

    return train_generator, test_generator

def build_model(num_classes):
    """
    Build the neural network using transfer learning.

    TRANSFER LEARNING:
    - Start with MobileNetV2 (trained on 1.4M images)
    - Keep its knowledge about shapes, edges, colors
    - Add our own layers to classify our 9 fruit classes

    Architecture:
        Input (224x224x3)
            |
        MobileNetV2 Base (FROZEN)
            |
        GlobalAveragePooling
            |
        Dense(256) + ReLU
            |
        Dropout(0.5)
            |
        Dense(num_classes) + Softmax
            |
        Output (9 probabilities)

    Args:
        num_classes: Number of fruit categories (9)

    Returns:
        model: Compiled Keras model ready to train
    """
    print("Building model architecture...")

    # Load pre-trained MobileNetV2
    # WHY MobileNetV2?
    # - Fast and lightweight (good for mobile)
    # - Excellent accuracy
    # - Only 31 MB
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),  # 224x224 RGB
        include_top=False,                     # Remove original classifier
        weights='imagenet'                     # Use ImageNet weights
    )

    # Freeze base model
    # WHY? Keep ImageNet knowledge, only train our new layers
    base_model.trainable = False

    print(f"  Base: MobileNetV2 (frozen)")
    print(f"  Base parameters: {base_model.count_params():,}")

    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)         # Reduce dimensions
    x = Dense(256, activation='relu')(x)    # Learn fruit features
    x = Dropout(0.5)(x)                     # Prevent overfitting
    outputs = Dense(num_classes, activation='softmax')(x)  # 9 outputs

    # Create final model
    model = Model(inputs=base_model.input, outputs=outputs)

    print(f"  Custom layers: GlobalPooling + Dense(256) + Dropout + Dense({num_classes})")
    print(f"  Total parameters: {model.count_params():,}")

    # Compile model
    # Optimizer: Adam (adaptive learning rate)
    # Loss: categorical_crossentropy (for multi-class)
    # Metrics: accuracy (easy to understand)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Model built and compiled")

    return model

def plot_training_history(history, save_path):
    """
    Create graphs showing training progress.

    Graphs:
    - Accuracy over epochs (should increase)
    - Loss over epochs (should decrease)

    Args:
        history: Training history from model.fit()
        save_path: Where to save the graph image
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(history.history['loss'], label='Training', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training graphs saved: {save_path}")

def save_training_artifacts(model, train_generator, history, final_acc, final_loss):
    """
    Save all training outputs.

    Saves:
    - Trained model (.keras file)
    - Class labels (JSON)
    - Training configuration (JSON)
    - Training graphs (PNG)

    Args:
        model: Trained Keras model
        train_generator: Training data generator (for class names)
        history: Training history
        final_acc: Final validation accuracy
        final_loss: Final validation loss
    """
    print("Saving training artifacts...")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Save model
    model_path = os.path.join(OUTPUT_DIR, "fruit_classifier.keras")
    model.save(model_path)
    print(f"Model saved: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")

    # 2. Save class labels
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    labels_path = os.path.join(OUTPUT_DIR, "class_labels.json")
    with open(labels_path, 'w') as f:
        json.dump(class_labels, f, indent=2)
    print(f"Class labels saved: {labels_path}")

    # 3. Save training config
    config = {
        "training_date": datetime.now().isoformat(),
        "model_architecture": "MobileNetV2 + Custom Head",
        "image_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_classes": len(class_labels),
        "training_samples": train_generator.samples,
        "final_val_accuracy": float(final_acc),
        "final_val_loss": float(final_loss),
        "class_names": list(class_labels.values())
    }
    config_path = os.path.join(OUTPUT_DIR, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved: {config_path}")

    # 4. Save training graphs
    graph_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plot_training_history(history, graph_path)

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    """
    Main training pipeline.

    Steps:
    1. Setup and configuration
    2. Load data
    3. Build model
    4. Train model
    5. Evaluate results
    6. Save everything
    """

    # Print welcome message
    print_section("FRUIT RIPENESS CLASSIFIER - TRAINING")
    print("Student: Maria Paula Salazar Agudelo")
    print("Course: AI Minor - Personal Challenge")
    print()

    # Record start time
    start_time = time.time()

    # ========================================================================
    # STEP 1: Environment Check
    # ========================================================================
    print_section("STEP 1: Environment Check")

    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus[0].name}")
        print("Training will be FAST (2-3 min/epoch)")
    else:
        print("No GPU detected - using CPU")
        print("Training will be SLOW (15-20 min/epoch)")

    # Verify dataset paths
    if not os.path.exists(TRAIN_DIR):
        print(f"\nERROR: Training directory not found!")
        print(f"Looking for: {TRAIN_DIR}")
        print(f"\nPlease update DATA_ROOT in this script to point to your dataset.")
        return

    if not os.path.exists(TEST_DIR):
        print(f"\nERROR: Test directory not found!")
        print(f"Looking for: {TEST_DIR}")
        print(f"\nPlease update DATA_ROOT in this script to point to your dataset.")
        return

    print(f"Dataset found: {DATA_ROOT}")

    # ========================================================================
    # STEP 2: Load Data
    # ========================================================================
    print_section("STEP 2: Load and Prepare Data")

    train_generator, test_generator = create_data_generators()

    num_classes = len(train_generator.class_indices)
    print(f"\nClasses ({num_classes}):")
    for name, idx in sorted(train_generator.class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx}. {name}")

    # ========================================================================
    # STEP 3: Build Model
    # ========================================================================
    print_section("STEP 3: Build Neural Network")

    model = build_model(num_classes)

    # Display model summary
    print("\nModel Summary:")
    model.summary()

    # ========================================================================
    # STEP 4: Train Model
    # ========================================================================
    print_section("STEP 4: Train the Model")

    print(f"Training for {EPOCHS} epochs...")
    print(f"Estimated time: {EPOCHS * 3} - {EPOCHS * 5} minutes with GPU")
    print(f"                {EPOCHS * 15} - {EPOCHS * 20} minutes with CPU")
    print()
    print("What to watch:")
    print("  - accuracy and val_accuracy should INCREASE")
    print("  - loss and val_loss should DECREASE")
    print("  - val_accuracy should be close to accuracy (no overfitting)")
    print()

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        verbose=1  # Show progress bar
    )

    # ========================================================================
    # STEP 5: Evaluate Results
    # ========================================================================
    print_section("STEP 5: Evaluate Results")

    # Get final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    print("FINAL RESULTS:")
    print(f"  Training Accuracy:   {final_train_acc*100:.2f}%")
    print(f"  Validation Accuracy: {final_val_acc*100:.2f}%")
    print(f"  Training Loss:       {final_train_loss:.4f}")
    print(f"  Validation Loss:     {final_val_loss:.4f}")
    print()

    # Interpretation
    if final_val_acc >= 0.85:
        print("EXCELLENT! Model meets project goal (>=85% accuracy)")
    elif final_val_acc >= 0.70:
        print("GOOD! Model works well (70-85% accuracy)")
    else:
        print("FAIR - Model could be improved (<70% accuracy)")

    # Check for overfitting
    if final_train_acc - final_val_acc > 0.10:
        print("\nWarning: Possible overfitting detected")
        print("Training accuracy much higher than validation")
        print("Model might be memorizing instead of learning")
    else:
        print("\nNo overfitting detected")
        print("Model generalizes well to new images")

    # ========================================================================
    # STEP 6: Save Everything
    # ========================================================================
    print_section("STEP 6: Save Model and Results")

    save_training_artifacts(model, train_generator, history, final_val_acc, final_val_loss)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section("TRAINING COMPLETE!")

    # Calculate total time
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print()
    print("Files created:")
    print(f"  models/fruit_classifier.keras (trained model)")
    print(f"  models/class_labels.json (class names)")
    print(f"  models/training_config.json (training info)")
    print(f"  models/training_history.png (training graphs)")
    print()
    print("Next steps:")
    print("  1. Run: python scripts/predict.py (to test predictions)")
    print("  2. Open: notebooks/03_Model_Evaluation.ipynb (detailed analysis)")
    print("  3. Run: python app.py (web interface with camera)")
    print()
    print("="*70)

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Model state not saved")
    except Exception as e:
        print(f"\n\nERROR during training: {str(e)}")
        print("\nIf you see this error, please:")
        print("  1. Check that dataset path is correct")
        print("  2. Verify TensorFlow is installed: pip install tensorflow")
        print("  3. Make sure you have enough disk space")
        raise
