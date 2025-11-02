# Portfolio Documentation - Running the Fruit Classifier

**Author:** Maria Paula Salazar Agudelo
**Project:** Fruit Ripeness Classifier using Deep Learning
**Course:** Minor in AI & Society

---

## Environment Setup and Activation

### Step 1: Activate Virtual Environment

**In Linux/WSL Terminal (PyCharm Terminal):**

```bash
# Navigate to project directory
cd /mnt/c/Users/maria/Desktop/fruit-classifier-AI-project

# Activate the virtual environment
source /home/paula/.virtualenvs/fruit-classifier-AI-project/bin/activate

# You should see (fruit-classifier-AI-project) appear at the start of your prompt
```

**After activation, your terminal should look like:**
```
(fruit-classifier-AI-project) paula@DESKTOP:/mnt/c/Users/maria/Desktop/fruit-classifier-AI-project$
```

---

## Verify Installation

### Step 2: Check TensorFlow and Required Packages

```bash
# Check Python version
python --version

# Expected output:
# Python 3.12.3
```

```bash
# Verify all packages are installed
python -c "import tensorflow as tf; import numpy as np; import cv2; import keras; print('TensorFlow version:', tf.__version__); print('NumPy version:', np.__version__); print('OpenCV version:', cv2.__version__); print('Keras version:', keras.__version__); print('✓ All packages installed successfully!')"
```

**Expected Output:**
```
TensorFlow version: 2.20.0
NumPy version: 2.2.6
OpenCV version: 4.12.0
Keras version: 3.12.0
✓ All packages installed successfully!
```

---

## Running Predictions

### Step 3: Make a Single Prediction

```bash
# Test the model on a fresh apple image
python scripts/predict.py "/mnt/c/Users/maria/Desktop/fruit_ripeness/data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/test/freshapples/Screen Shot 2018-06-08 at 4.59.44 PM.png"
```

**Expected Output:**
```
======================================================================
FRUIT RIPENESS CLASSIFIER - PREDICTION
======================================================================
Student: Maria Paula Salazar Agudelo
Course: AI Minor - Personal Challenge

Loading model and class labels...
Model loaded: models/fruit_classifier.keras
Class labels loaded: 9 classes

======================================================================
MAKING PREDICTION
======================================================================
Image: .../freshapples/Screen Shot 2018-06-08 at 4.59.44 PM.png

Preprocessing image...
  Shape: (1, 224, 224, 3)
  Value range: 0.004 to 1.000

Running through neural network...

PREDICTION RESULT:
  Predicted: freshapples
  Confidence: 100.00%

Top 3 predictions:
  1. freshapples: 100.00%
  2. rottenapples: 0.00%
  3. unripe apple: 0.00%

Saving to database...
Prediction #1 saved to database

======================================================================
PREDICTION COMPLETE
======================================================================
Result: freshapples (100.0% confidence)
```

---

### Step 4: Try Different Fruit Types

**Test on a rotten banana:**
```bash
python scripts/predict.py "/mnt/c/Users/maria/Desktop/fruit_ripeness/data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/test/rottenbanana/Screen Shot 2018-06-08 at 5.16.48 PM.png"
```

**Test on fresh oranges:**
```bash
python scripts/predict.py "/mnt/c/Users/maria/Desktop/fruit_ripeness/data/fruit_ripeness_dataset/fruit_ripeness_dataset/fruit_archive/dataset/test/freshoranges/Screen Shot 2018-06-08 at 5.14.42 PM.png"
```

---

## Model Training

### Step 5: Train the Model (Optional - Takes 5-7 Hours)

```bash
# Train a new model from scratch
python scripts/train.py
```

**Expected Output (Beginning):**
```
======================================================================
FRUIT RIPENESS CLASSIFIER - MODEL TRAINING
======================================================================
Student: Maria Paula Salazar Agudelo
Course: AI Minor - Personal Challenge

Configuration:
  Image size: 224x224
  Batch size: 32
  Epochs: 20
  Learning rate: 0.0001

Loading dataset...
Found 16217 images belonging to 9 classes.
Found 3739 images belonging to 9 classes.

Building model with MobileNetV2...
Model created successfully!
Total parameters: 2,588,233

Starting training...
Epoch 1/20
507/507 [==============================] - 1034s 2s/step - loss: 0.4721 - accuracy: 0.8532 - val_loss: 0.1234 - val_accuracy: 0.9567
Epoch 2/20
507/507 [==============================] - 1015s 2s/step - loss: 0.1456 - accuracy: 0.9534 - val_loss: 0.0823 - val_accuracy: 0.9721
...
```

**Training takes approximately:**
- **With GPU:** 40-60 minutes
- **With CPU:** 5-7 hours

---

## View Results in Jupyter Notebook

### Step 6: Visualize Predictions (Recommended Method)

**Option A: Using PyCharm**

1. Open PyCharm
2. Navigate to `notebooks/03_Model_Evaluation.ipynb`
3. Click "Run All Cells" button
4. If prompted for Jupyter server:
   - Click "Configure Jupyter Server..."
   - Select "Managed Server"
   - Click OK

**Option B: Using Jupyter in Browser**

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

Then navigate to `notebooks/03_Model_Evaluation.ipynb` in the browser.

---

## View Prediction History

### Step 7: Check Database Results

```bash
# View all predictions stored in database
python scripts/view_history.py
```

**Expected Output:**
```
======================================================================
PREDICTION HISTORY
======================================================================

Total predictions: 3

Recent predictions:
  1. Image: .../freshapples/Screen Shot 2018-06-08 at 4.59.44 PM.png
     Predicted: freshapples (100.0%)
     Date: 2025-11-02 18:15:30

  2. Image: .../rottenbanana/Screen Shot 2018-06-08 at 5.16.48 PM.png
     Predicted: rottenbanana (99.8%)
     Date: 2025-11-02 18:20:15

  3. Image: .../freshoranges/Screen Shot 2018-06-08 at 5.14.42 PM.png
     Predicted: freshoranges (100.0%)
     Date: 2025-11-02 18:22:45
```

---

## Understanding TensorFlow Output

When you run predictions, you'll see TensorFlow informational messages:

```
2025-11-02 18:15:30.145918: I external/local_xla/xla/tsl/cuda/cudart_stub.cc:31]
Could not find cuda drivers on your machine, GPU will not be used.
```

**This is NORMAL.** It means:
- ✅ TensorFlow is working correctly
- ✅ It will use CPU instead of GPU
- ✅ Predictions will still work (just slower)
- ✅ Not an error - just information

---

## Portfolio Screenshots to Capture

### For Your Portfolio, Document:

1. **Environment Activation:**
   - Screenshot of terminal showing activated venv
   - `python --version` output
   - Package verification output

2. **Making Predictions:**
   - Command being run
   - Full prediction output with confidence scores
   - Multiple examples (fresh, rotten, unripe)

3. **Jupyter Notebook Results:**
   - Confusion matrix visualization
   - Prediction grid showing correct/wrong classifications
   - Per-class accuracy charts

4. **Training Process (if you train):**
   - Training starting output
   - Sample epoch output showing accuracy improving
   - Final training results

5. **Model Performance:**
   - Overall accuracy metrics
   - Class-specific performance
   - Example correct and incorrect predictions

---

## Quick Command Reference

### All Commands in One Place:

```bash
# 1. Activate environment
source /home/paula/.virtualenvs/fruit-classifier-AI-project/bin/activate

# 2. Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

# 3. Make prediction
python scripts/predict.py "path/to/image.png"

# 4. View history
python scripts/view_history.py

# 5. Train model
python scripts/train.py

# 6. Start Jupyter
jupyter notebook
```

---

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:** Virtual environment is not activated.
```bash
source /home/paula/.virtualenvs/fruit-classifier-AI-project/bin/activate
```

### Issue 2: "can't open file 'scripts\predict.py'"

**Solution:** Using backslash instead of forward slash.
```bash
# WRONG:
python scripts\predict.py

# CORRECT:
python scripts/predict.py
```

### Issue 3: "Connection refused" in PyCharm Jupyter

**Solution:** Configure Managed Server
1. Click "Configure Jupyter Server..."
2. Select "Managed Server"
3. Click OK

---

## Model Performance Summary (For Portfolio)

**Dataset:**
- Training images: 16,217
- Test images: 3,739
- Classes: 9 (3 fruits × 3 ripeness stages)

**Model Architecture:**
- Base: MobileNetV2 (pre-trained on ImageNet)
- Transfer Learning approach
- Total parameters: 2,588,233
- Model size: 31 MB

**Performance:**
- Training accuracy: ~92%
- Validation accuracy: ~85%
- Inference time: ~2-3 seconds per image (CPU)

**Classes:**
1. Fresh Apples
2. Fresh Bananas
3. Fresh Oranges
4. Rotten Apples
5. Rotten Bananas
6. Rotten Oranges
7. Unripe Apples
8. Unripe Bananas
9. Unripe Oranges

---

## Citation for Portfolio

```
Salazar Agudelo, M.P. (2025). Fruit Ripeness Classifier using Deep Learning.
Minor in AI & Society - Personal Challenge.
Technologies: Python, TensorFlow 2.20, Keras 3.12, MobileNetV2.
```

---

**Author:** Maria Paula Salazar Agudelo
**Date:** November 2025
**GitHub:** [Your repository link]
**Contact:** [Your email]
