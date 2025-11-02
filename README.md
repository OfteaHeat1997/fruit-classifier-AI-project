# Fruit Ripeness Classifier

**Author:** Maria Paula Salazar Agudelo
**Course:** Minor in AI & Society
**Project:** Personal Challenge - Fruit Ripeness Detection

---

## Overview

A deep learning system that classifies fruit images by ripeness stage. The model can recognize **9 categories**:

**Fruits:** Apples, Bananas, Oranges
**Ripeness stages:** Fresh, Rotten, Unripe

### Project Goal

Build a mobile-first tool to help shoppers buy better fruit by classifying ripeness from a camera photo.

**Target accuracy:** ≥ 85%

---

## Technology Stack

- **Framework:** TensorFlow / Keras
- **Model:** MobileNetV2 (transfer learning)
- **Dataset:** ~20,000 fruit images
- **Language:** Python 3.8+

---

## Project Structure

```
fruit-classifier-AI-project/
├── notebooks/
│   ├── 01_Dataset_Analysis.ipynb      # Dataset exploration
│   ├── 02_Model_Training.ipynb        # Model training (current)
│   └── 03_Model_Evaluation.ipynb      # Performance analysis (coming soon)
├── scripts/
│   ├── train.py                       # Training script
│   └── predict.py                     # Prediction script
├── models/
│   ├── fruit_classifier.keras         # Trained model
│   ├── class_labels.json              # Class names
│   └── training_config.json           # Training parameters
├── README.md
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/OfteaHeat1997/fruit-classifier-AI-project.git
cd fruit-classifier-AI-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download dataset

Dataset used: Fruit Ripeness Dataset from Kaggle
- Place in: `data/` folder
- Structure: `data/train/` and `data/test/` with 9 class folders

---

## Usage

### Train the model

```bash
python scripts/train.py
```

### Make predictions

```bash
python scripts/predict.py path/to/image.jpg
```

### Run notebooks

```bash
jupyter notebook notebooks/02_Model_Training.ipynb
```

---

## Model Architecture

**Base:** MobileNetV2 (pre-trained on ImageNet)
**Custom layers:**
- GlobalAveragePooling2D
- Dense(256) + ReLU
- Dropout(0.5)
- Dense(9) + Softmax

**Parameters:** ~2.6 million
**Model size:** ~3 MB

---

## Training Details

- **Epochs:** 20
- **Batch size:** 32
- **Learning rate:** 0.0001
- **Data augmentation:** Rotation, flip, zoom, shift
- **Training time:** ~40-60 minutes (GPU) / 5-7 hours (CPU)

---

## Results

- **Validation accuracy:** ~85%+
- **Training samples:** ~16,000 images
- **Test samples:** ~3,700 images

See `notebooks/02_Model_Training.ipynb` for detailed results and graphs.

---

## Future Work

1. Convert to TensorFlow Lite for mobile deployment
2. Build Flask API for web access
3. Create Flutter mobile application
4. Add more fruit types and ripeness stages
5. Implement model quantization for faster inference

---

## Portfolio Deliverables

### Phase 1 - Dataset Analysis ✅
- Dataset exploration and visualization
- Class distribution analysis
- Quality inspection

### Phase 2 - Model Training ✅ (Current)
- Transfer learning with MobileNetV2
- Training with data augmentation
- Model evaluation and saving

### Phase 3 - Deployment (Coming Soon)
- Python demo (Flask/Streamlit)
- TensorFlow Lite conversion
- Flutter mobile app

---

## License

Educational project for Minor in AI & Society course.

---

## Contact

**Student:** Maria Paula Salazar Agudelo
**GitHub:** [@OfteaHeat1997](https://github.com/OfteaHeat1997)
