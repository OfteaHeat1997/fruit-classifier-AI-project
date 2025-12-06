# Tool Selection Research

**Student:** Maria Paula Salazar Agudelo
**Course:** Minor in AI & Society - Personal Challenge
**Purpose:** Document research and decisions for tool/technology choices

---

## Why This Document Matters

For my portfolio, I need to show that I made **informed decisions** - not just "I used TensorFlow because a tutorial said so."

This document shows:
- What options I considered
- How I evaluated them
- Why I chose what I chose
- Trade-offs I accepted

---

## Table of Contents

1. [Deep Learning Framework](#1-deep-learning-framework)
2. [Pre-trained Model Architecture](#2-pre-trained-model-architecture)
3. [Transfer Learning vs Training from Scratch](#3-transfer-learning-vs-training-from-scratch)
4. [Programming Language](#4-programming-language)
5. [Dataset Selection](#5-dataset-selection)
6. [Development Environment](#6-development-environment)
7. [Summary Decision Matrix](#7-summary-decision-matrix)

---

## 1. Deep Learning Framework

### Options Considered

| Framework | Pros | Cons |
|-----------|------|------|
| **TensorFlow/Keras** | Industry standard, excellent documentation, easy deployment, Google support | Can be verbose, steep learning curve for advanced features |
| **PyTorch** | Pythonic, flexible, popular in research | Deployment more complex, less beginner-friendly |
| **FastAI** | Very high-level, quick prototyping | Less control, smaller community |
| **JAX** | Very fast, functional programming | Very new, steep learning curve |

### My Research

**TensorFlow/Keras:**
- Used by Google, Airbnb, Twitter, Intel
- Keras is high-level API - easier for beginners
- TensorFlow Lite for mobile deployment (important for my project!)
- Huge community = easy to find help
- Source: https://www.tensorflow.org/about

**PyTorch:**
- Used by Facebook, Tesla, Microsoft
- More popular in academic research
- Dynamic computation graphs (more flexible)
- Source: https://pytorch.org/

### Decision: TensorFlow/Keras

**Why I chose it:**

1. **Mobile Deployment:** My project goal is a mobile app. TensorFlow Lite makes this easy:
   ```
   TensorFlow → TensorFlow Lite → Mobile App
   ```
   PyTorch requires extra steps (ONNX conversion).

2. **Beginner-Friendly:** Keras API is intuitive:
   ```python
   # Keras - easy to read
   model = Sequential([
       Dense(256, activation='relu'),
       Dropout(0.5),
       Dense(9, activation='softmax')
   ])
   ```

3. **Documentation:** Official tutorials matched my use case (image classification).

4. **Industry Standard:** If I learn TensorFlow, it's useful for jobs.

**Trade-offs I Accepted:**
- PyTorch might be more flexible for research
- But I'm building a product, not doing research

---

## 2. Pre-trained Model Architecture

### Options Considered

| Model | Size | Top-1 Accuracy (ImageNet) | Parameters | Speed |
|-------|------|---------------------------|------------|-------|
| **MobileNetV2** | 13 MB | 71.8% | 3.4M | Very Fast |
| **ResNet50** | 98 MB | 76.0% | 25.6M | Medium |
| **VGG16** | 528 MB | 71.3% | 138M | Slow |
| **EfficientNetB0** | 29 MB | 77.1% | 5.3M | Fast |
| **InceptionV3** | 92 MB | 77.9% | 23.8M | Medium |

### My Research

**MobileNetV2:**
- Paper: "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018)
- Designed specifically for mobile devices
- Uses "depthwise separable convolutions" - same accuracy, fewer computations
- Source: https://arxiv.org/abs/1801.04381

**ResNet50:**
- Paper: "Deep Residual Learning" (He et al., 2015)
- Very accurate but large (98 MB)
- "Skip connections" solve vanishing gradient problem
- Source: https://arxiv.org/abs/1512.03385

**VGG16:**
- Classic architecture, very simple
- BUT: 528 MB is way too big for mobile
- Source: https://arxiv.org/abs/1409.1556

### Decision: MobileNetV2

**Why I chose it:**

1. **Size for Mobile:** 13 MB vs 98+ MB for others
   ```
   Phone storage is limited
   Users won't download 100MB app for fruit classification
   ```

2. **Speed:** Designed for real-time mobile inference
   - Can process images in <100ms on phone
   - Users expect instant results when taking photo

3. **Accuracy Trade-off is Worth It:**
   - MobileNetV2: 71.8% on ImageNet (1000 classes)
   - My task: Only 9 classes (much easier!)
   - Result: 97.3% accuracy on my specific task

4. **TensorFlow Lite Optimized:** MobileNetV2 works great with TFLite quantization

**Trade-offs I Accepted:**
- ResNet50 might give 1-2% higher accuracy
- But 7x larger model isn't worth it for mobile

**Visual Comparison:**
```
Model Size Comparison:
MobileNetV2  ████ 13 MB
EfficientNet ████████ 29 MB
ResNet50     ████████████████████████████████ 98 MB
VGG16        ████████████████████████████████████████████████████████ 528 MB
             ↑
             My choice (smallest that's still accurate)
```

---

## 3. Transfer Learning vs Training from Scratch

### Options Considered

| Approach | Training Time | Data Needed | Accuracy |
|----------|---------------|-------------|----------|
| **Transfer Learning** | Hours | 10K-50K images | High |
| **Train from Scratch** | Days/Weeks | 100K+ images | Medium-High |
| **Use Pre-trained Only** | Minutes | 0 | Low (wrong classes) |

### My Research

**Transfer Learning Concept:**
- Use model pre-trained on ImageNet (1.4 million images)
- Keep the "learned features" (edges, shapes, textures)
- Only retrain final classification layers

**Why it Works:**
```
ImageNet Model knows:
├── Low-level: Edges, corners, gradients
├── Mid-level: Textures, patterns, shapes
└── High-level: Object parts

My Fruits share these features!
├── Apples have round shapes (ImageNet knows "round")
├── Bananas have yellow color (ImageNet knows "yellow")
└── Rotten fruit has brown spots (ImageNet knows "spots")
```

**Source:**
- "How transferable are features in deep neural networks?" (Yosinski et al., 2014)
- https://arxiv.org/abs/1411.1792

### Decision: Transfer Learning (Freeze Base + Train Head)

**Why I chose it:**

1. **Limited Data:** I have 16,217 training images
   - Sounds like a lot, but not for training from scratch
   - ImageNet models trained on 1.4 MILLION images

2. **Limited Compute:** No GPU, only CPU
   - Training from scratch: Days or weeks
   - Transfer learning: 6 hours on CPU

3. **Proven Results:**
   - First epoch already 85% accuracy!
   - From scratch would start at ~11% (random guessing for 9 classes)

**What I Actually Did:**
```
MobileNetV2 Base (2.2M parameters) → FROZEN (keep ImageNet knowledge)
                ↓
    Custom Head (400K parameters) → TRAINED (learn fruit classes)
                ↓
         Result: 97.3% accuracy
```

**Trade-offs I Accepted:**
- Model might not learn fruit-specific low-level features
- But 97.3% accuracy shows this wasn't needed

---

## 4. Programming Language

### Options Considered

| Language | ML Support | Learning Curve | Job Market |
|----------|------------|----------------|------------|
| **Python** | Excellent | Easy | Very High |
| **R** | Good (statistics) | Medium | Medium |
| **Julia** | Growing | Medium | Low |
| **JavaScript** | Limited | Easy | High (web) |

### Decision: Python

**Why I chose it:**

1. **ML Standard:** TensorFlow, PyTorch, scikit-learn all Python-first
2. **Libraries:** NumPy, Pandas, Matplotlib, OpenCV
3. **Community:** Most tutorials, Stack Overflow answers in Python
4. **Career:** Most ML job postings require Python

**No real alternatives for this project** - Python is the clear winner for ML.

---

## 5. Dataset Selection

### Options Considered

| Dataset | Images | Classes | Source | DOI |
|---------|--------|---------|--------|-----|
| **Kaggle Fruit Ripeness** | 20K | 9 | Kaggle | No |
| **Mendeley Fresh/Rotten** | 13K | 16 | Academic | Yes |
| **Fruits-360** | 90K | 131 | Academic | Yes |
| **Collect Own Data** | ? | ? | Self | N/A |

### My Research

**Kaggle Dataset:**
- Easy to download
- Already split into train/test
- Good quality images
- BUT: No DOI, no academic paper, unknown collection method

**Mendeley Dataset:**
- DOI: 10.17632/bdd69gyhv8.1
- Published paper describes collection method
- Verified by domain experts
- Academic credibility for portfolio

**Source:**
- Mendeley: https://data.mendeley.com/datasets/bdd69gyhv8/1
- Fruits-360 Paper: https://doi.org/10.2478/ausi-2018-0002

### Decision: Started with Kaggle, Adding Academic Sources

**Why:**

1. **Initial Choice (Kaggle):** Quick start, good quality
2. **Portfolio Requirement:** Need academic provenance (LO3)
3. **Solution:** Document Kaggle usage, add Mendeley for credibility

**Trade-offs I Accepted:**
- Kaggle data works well but lacks provenance
- Academic datasets have provenance but may need more preprocessing

---

## 6. Development Environment

### Options Considered

| Environment | GPU Access | Cost | Setup |
|-------------|------------|------|-------|
| **Local (WSL)** | No (CPU only) | Free | Medium |
| **Google Colab** | Yes (free tier) | Free | Easy |
| **Kaggle Notebooks** | Yes (30h/week) | Free | Easy |
| **AWS/GCP** | Yes | Paid | Complex |

### Decision: Local WSL (with Colab backup option)

**Why I chose it:**

1. **Learning:** Setting up local environment teaches real skills
2. **No Dependencies:** Don't need internet during training
3. **Full Control:** Can customize everything

**Trade-offs I Accepted:**
- Training takes 6+ hours on CPU
- Could be 30 minutes on Colab GPU
- But I learn more about the process by waiting and watching

**For Future:** Would use Colab for experiments, local for final training

---

## 7. Summary Decision Matrix

| Decision | Chosen | Main Reason | Alternative |
|----------|--------|-------------|-------------|
| **Framework** | TensorFlow/Keras | Mobile deployment (TFLite) | PyTorch |
| **Architecture** | MobileNetV2 | Small size (13MB) for mobile | ResNet50 |
| **Approach** | Transfer Learning | Limited data & compute | Train from scratch |
| **Language** | Python | Industry standard for ML | None viable |
| **Dataset** | Kaggle + Academic | Quality + Provenance | Own collection |
| **Environment** | Local WSL | Learning experience | Google Colab |

---

## Reflection: What I Learned

### About Making Decisions

1. **No Perfect Choice:** Every tool has trade-offs
2. **Context Matters:** MobileNetV2 is best FOR MY USE CASE (mobile)
3. **Research First:** Reading papers helped me understand WHY tools work

### What I Would Do Differently

1. **Start with Colab:** For faster experimentation
2. **Academic Data First:** Would save time on provenance later
3. **Document Earlier:** Writing this doc helped clarify my thinking

### For Portfolio Defense

**If asked "Why TensorFlow?"**
> "I chose TensorFlow because my goal is mobile deployment. TensorFlow Lite makes converting models for Android/iOS straightforward. PyTorch would require additional conversion steps through ONNX."

**If asked "Why MobileNetV2?"**
> "MobileNetV2 is designed for mobile devices - only 13MB compared to 98MB for ResNet50. For a mobile app where users need quick results, the speed and size advantages outweigh the small accuracy difference."

**If asked "Why Transfer Learning?"**
> "With only 16,000 training images and no GPU, training from scratch would take weeks and likely achieve worse results. Transfer learning lets me use ImageNet's knowledge of visual features, achieving 97.3% accuracy in just 6 hours of training."

---

## References

1. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." arXiv:1801.04381
2. He, K., et al. (2015). "Deep Residual Learning for Image Recognition." arXiv:1512.03385
3. Yosinski, J., et al. (2014). "How transferable are features in deep neural networks?" arXiv:1411.1792
4. TensorFlow Documentation: https://www.tensorflow.org/
5. Keras Documentation: https://keras.io/

---

**Last Updated:** December 2025
**Status:** Complete

---

*This document demonstrates that my tool choices were researched and justified, not arbitrary. Each decision considered alternatives and accepted specific trade-offs based on project requirements.*
