# FreshScan - Portfolio Summary

**Project:** Fruit Ripeness Classifier
**Author:** Maria Paula Salazar Agudelo
**Program:** Minor in AI & Society - Fontys
**Date:** December 2025
**MMM Label:** AI-Assisted

---

## Project Overview

FreshScan is a mobile application that uses AI to classify fruit ripeness, helping Dutch consumers reduce food waste by making informed decisions about fruit freshness.

### Problem Statement
- **88 million tonnes** of food wasted annually in the EU
- Consumers cannot accurately judge fruit ripeness
- Fruits are among the most wasted food categories

### Solution
A CNN-based mobile app that:
1. Takes a photo of fruit
2. Analyzes ripeness using AI (MobileNetV2)
3. Provides actionable recommendations (eat now, store, compost)

---

## Learning Outcomes Demonstrated

### LO1: Data Management
- Collected and organized 19,956 images
- Implemented train/test split (80/20)
- Applied data augmentation techniques

### LO2: AI System Architecture
- Transfer learning with MobileNetV2
- TensorFlow/Keras implementation
- Flask REST API for deployment

### LO3: Data Understanding (Key Focus)
| Requirement | Implementation |
|-------------|----------------|
| Data provenance | DOI: 10.17632/bdd69gyhv8.1 (Mendeley) |
| Academic source | Sultana et al., Data in Brief 2022 |
| Collection method | Documented in peer-reviewed paper |
| License compliance | CC BY 4.0, MIT |
| Quality assessment | Validity, accuracy, completeness verified |

### LO4: Model Evaluation
- Validation accuracy: **97.3%**
- Confusion matrix analysis
- Per-class precision/recall

### LO5: Deployment
- React Native mobile app (Expo)
- Flask backend API
- TensorFlow Lite for mobile inference

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MOBILE APP                        │
│              (React Native / Expo)                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │  Home   │  │ Camera  │  │ Results │             │
│  └─────────┘  └─────────┘  └─────────┘             │
└───────────────────────┬─────────────────────────────┘
                        │ REST API
                        ▼
┌─────────────────────────────────────────────────────┐
│                  FLASK BACKEND                       │
│           /api/predict (POST)                        │
│           /api/health (GET)                          │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               AI MODEL (TFLite)                      │
│        MobileNetV2 + Custom Classification           │
│           9 Classes | 97.3% Accuracy                 │
└─────────────────────────────────────────────────────┘
```

---

## Data Provenance Summary

### Primary Dataset
| Attribute | Value |
|-----------|-------|
| Name | Fresh and Rotten Fruits |
| DOI | **10.17632/bdd69gyhv8.1** |
| Authors | Sultana, Jahan, Uddin |
| Institution | Jahangirnagar University |
| Published | Data in Brief, 2022 |
| Verification | Domain expert (Ministry of Agriculture) |

### Secondary Dataset (For Pears - European)
| Attribute | Value |
|-----------|-------|
| Name | Fruits-360 |
| DOI | **10.2478/ausi-2018-0002** |
| Authors | Muresan, Oltean |
| Institution | Babeș-Bolyai University, Romania |
| Relevance | European origin, contains pears |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Total Parameters | 2,588,233 |
| Training Samples | 16,217 |
| Test Samples | 3,739 |
| Validation Accuracy | 97.3% |
| Model Size | ~3 MB (Keras) / ~8 MB (TFLite) |

### Classes
1. Fresh Apple
2. Fresh Banana
3. Fresh Orange
4. Rotten Apple
5. Rotten Banana
6. Rotten Orange
7. Unripe Apple
8. Unripe Banana
9. Unripe Orange

---

## IBM Data Science Methodology

| Step | Implementation |
|------|----------------|
| 1. Business Understanding | Food waste reduction in Netherlands |
| 2. Analytic Approach | CNN image classification |
| 3. Data Requirements | Fruit images with ripeness labels |
| 4. Data Collection | Academic datasets with DOI |
| 5. Data Understanding | EDA, quality assessment, bias analysis |
| 6. Data Preparation | Augmentation, normalization, split |
| 7. Modeling | MobileNetV2 transfer learning |
| 8. Evaluation | 97.3% accuracy, confusion matrix |
| 9. Deployment | Mobile app + Flask API |

---

## DOT Framework Research

| Method | Application |
|--------|-------------|
| Literature Study | Dataset selection, CNN architectures |
| Prototyping | Mobile app development |
| Showroom | Demo to stakeholders |
| Field Research | Testing with real fruit photos |

---

## Files Structure

```
fruit-classifier-AI-project/
├── models/
│   ├── fruit_classifier.keras    # Trained model
│   ├── fruit_classifier.tflite   # Mobile model
│   ├── class_labels.json         # Class mapping
│   └── training_config.json      # Training parameters
├── notebooks/
│   ├── 00_AI_Methodology_Overview.ipynb
│   ├── 01_Dataset_Analysis.ipynb
│   ├── 02_Model_Training.ipynb
│   ├── 03_Model_Evaluation.ipynb
│   └── 04_Dataset_Merge_Balance.ipynb
├── mobile-app/                   # React Native app
├── webapp/                       # Flask backend
├── scripts/
│   ├── predict.py
│   ├── convert_to_tflite.py
│   └── download_datasets.py
├── DATA_PROVENANCE.md           # Full provenance documentation
├── DATA_REQUIREMENTS.md         # Data requirements justification
└── PORTFOLIO_SUMMARY.md         # This file
```

---

## Key Achievements

1. **Academic Data Sources** - Used datasets with DOI and peer-reviewed publications
2. **High Accuracy** - 97.3% validation accuracy
3. **Complete Pipeline** - From data collection to mobile deployment
4. **Documentation** - Full provenance and methodology documentation
5. **European Focus** - Included fruits common in Netherlands

---

## Future Improvements

1. Add more European fruits (pears, strawberries, grapes)
2. Collect own data at Dutch supermarkets
3. Implement on-device inference (no server needed)
4. Add nutritional information
5. Multi-language support (Dutch, English)

---

## References

1. Sultana N, et al. "An extensive dataset for successful recognition of fresh and rotten fruits." Data in Brief. 2022. DOI: 10.1016/j.dib.2022.108552

2. Muresan H, Oltean M. "Fruit recognition from images using deep learning." Acta Univ. Sapientiae. 2018. DOI: 10.2478/ausi-2018-0002

3. Howard AG, et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861. 2017.

---

*Portfolio prepared following Fontys AI curriculum requirements and IBM Data Science Methodology.*
