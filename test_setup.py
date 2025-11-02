"""
Test script to verify all packages and paths are working correctly.
This will help identify any remaining errors.

Run this with: venv/bin/python test_setup.py
"""

import os
import sys

print("="*70)
print("TESTING FRUIT CLASSIFIER SETUP")
print("="*70)
print()

# Test 1: Python version
print("TEST 1: Python Version")
print(f"  Python: {sys.version}")
print("  ✓ PASSED")
print()

# Test 2: Import all required packages
print("TEST 2: Package Imports")
try:
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
except Exception as e:
    print(f"  ✗ numpy FAILED: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"  ✓ tensorflow {tf.__version__}")
except Exception as e:
    print(f"  ✗ tensorflow FAILED: {e}")
    sys.exit(1)

try:
    from tensorflow import keras
    print(f"  ✓ keras {keras.__version__}")
except Exception as e:
    print(f"  ✗ keras FAILED: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"  ✓ matplotlib {matplotlib.__version__}")
except Exception as e:
    print(f"  ✗ matplotlib FAILED: {e}")
    sys.exit(1)

try:
    import PIL
    print(f"  ✓ pillow {PIL.__version__}")
except Exception as e:
    print(f"  ✗ pillow FAILED: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"  ✓ opencv-python {cv2.__version__}")
except Exception as e:
    print(f"  ✗ opencv-python FAILED: {e}")
    sys.exit(1)

print("  ✓ PASSED - All packages imported successfully")
print()

# Test 3: Dataset paths
print("TEST 3: Dataset Paths")
if os.name == 'nt':
    DATA_ROOT = r"C:\Users\maria\Desktop\fruit_ripeness\data\fruit_ripeness_dataset\fruit_ripeness_dataset\fruit_archive\dataset"
else:
    DATA_ROOT = "/mnt/c/Users/maria/Desktop/fruit_ripeness/data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset"

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

if not os.path.exists(DATA_ROOT):
    print(f"  ✗ FAILED: Dataset root not found: {DATA_ROOT}")
    sys.exit(1)
print(f"  ✓ Dataset root exists: {DATA_ROOT}")

if not os.path.exists(TRAIN_DIR):
    print(f"  ✗ FAILED: Train directory not found: {TRAIN_DIR}")
    sys.exit(1)
print(f"  ✓ Train directory exists")

if not os.path.exists(TEST_DIR):
    print(f"  ✗ FAILED: Test directory not found: {TEST_DIR}")
    sys.exit(1)
print(f"  ✓ Test directory exists")

train_classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
test_classes = [d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))]

print(f"  ✓ Training classes: {len(train_classes)}")
print(f"  ✓ Test classes: {len(test_classes)}")
print("  ✓ PASSED")
print()

# Test 4: TensorFlow/Keras imports
print("TEST 4: TensorFlow/Keras Model Components")
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    print("  ✓ All Keras components imported successfully")
    print("  ✓ PASSED")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)
print()

# Test 5: Data generators
print("TEST 5: Data Generators")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    print(f"  ✓ Training samples: {train_generator.samples}")
    print(f"  ✓ Test samples: {test_generator.samples}")
    print(f"  ✓ Number of classes: {len(train_generator.class_indices)}")
    print("  ✓ PASSED")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Test 6: Model creation
print("TEST 6: Model Creation")
try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(9, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"  ✓ Model created successfully")
    print(f"  ✓ Total parameters: {model.count_params():,}")
    print("  ✓ PASSED")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Final summary
print("="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
print()
print("Your environment is ready for training!")
print()
print("Next steps:")
print("  1. Run training script: venv/bin/python scripts/train.py")
print("  2. Or use Jupyter notebook: venv/bin/jupyter notebook")
print()
