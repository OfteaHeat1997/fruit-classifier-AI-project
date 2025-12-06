# Data Requirements & Justification Document

**Project:** FreshScan - Fruit Ripeness Classifier
**Author:** Maria Paula Salazar Agudelo
**Date:** December 2025
**MMM Label:** 🤖 AI-Assisted (Claude helped structure this document)

---

## 1. Project Context & Problem Definition

### 1.1 Problem Statement
Food waste is a significant problem in Europe and the Netherlands. According to the EU, approximately **88 million tonnes** of food are wasted annually, with fruits being one of the most wasted categories. Consumers often:
- Cannot accurately judge fruit ripeness
- Don't know optimal storage conditions
- Throw away edible fruit due to visual uncertainty

### 1.2 Target Users (Netherlands/Europe Focus)
| User Persona | Need | Context |
|--------------|------|---------|
| Dutch consumers | Know when fruit is ready to eat | Shopping at Albert Heijn, Jumbo |
| Families | Reduce household food waste | Weekly grocery shopping |
| Small grocers | Quality control | Local markets, toko's |

### 1.3 Research Question
> "How can a CNN-based mobile application help Dutch consumers reduce fruit waste by accurately classifying fruit ripeness?"

---

## 2. Data Requirements (LO3)

### 2.1 Functional Requirements

Based on the project context and target users in the Netherlands, the data must:

| Requirement | Justification | Priority |
|-------------|---------------|----------|
| **Include common Dutch/European fruits** | Apples, pears, oranges, bananas, strawberries, grapes are most consumed in NL | Must Have |
| **Cover ripeness stages** | Fresh, Unripe, Overripe/Rotten for actionable advice | Must Have |
| **High-quality images** | Mobile app will use phone cameras; need similar quality | Must Have |
| **Balanced classes** | Prevent model bias toward majority class | Must Have |
| **Diverse backgrounds** | Real-world usage won't have studio conditions | Should Have |
| **Multiple angles** | Users may photograph from different positions | Should Have |

### 2.2 Non-Functional Requirements

| Requirement | Specification | Reason |
|-------------|---------------|--------|
| Image size | Min 224x224 pixels | MobileNetV2 input requirement |
| Format | JPG/PNG | Standard mobile formats |
| Minimum samples per class | 1000+ images | Statistical significance |
| Lighting conditions | Various (natural, artificial) | Real-world robustness |

---

## 3. Dataset Selection & Justification

### 3.1 Selection Criteria (DOT Framework - Literature Study)

I evaluated datasets using these **Data Quality Dimensions**:

| Dimension | Question | Weight |
|-----------|----------|--------|
| **Validity** | Does data represent what we need? | High |
| **Completeness** | Are all required classes present? | High |
| **Accuracy** | Are labels correct? | High |
| **Timeliness** | Is data recent enough? | Medium |
| **Consistency** | Same format/quality throughout? | Medium |

### 3.2 Datasets Evaluated

#### Dataset 1: Fruits Fresh and Rotten (Kaggle)
- **Source:** https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
- **Contents:** Fresh/Rotten: Apples, Bananas, Oranges
- **Size:** ~13,000 images
- **Pros:** ✅ Matches our core fruits, ✅ Good quality, ✅ Balanced
- **Cons:** ❌ No unripe category, ❌ No European fruits (pears, strawberries)
- **Decision:** ✅ USE - Primary dataset for core classes

#### Dataset 2: Fruit Ripeness: Unripe, Ripe, Rotten (Kaggle)
- **Source:** https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten
- **Contents:** Three ripeness stages for tropical fruits
- **Pros:** ✅ Has unripe category, ✅ Multiple stages
- **Cons:** ❌ Mostly tropical fruits
- **Decision:** ✅ USE - For unripe class augmentation

#### Dataset 3: Extensive Fresh/Rotten Dataset (Mendeley)
- **Source:** https://data.mendeley.com/datasets/bdd69gyhv8/1
- **Contents:** 16 fruit types including strawberries, grapes
- **Size:** Large dataset with augmentation
- **Pros:** ✅ Includes strawberries & grapes, ✅ Academic source
- **Cons:** ❌ No pears
- **Decision:** ✅ USE - For expanding fruit variety

#### Dataset 4: Fruits-360 (Kaggle/GitHub)
- **Source:** https://github.com/Horea94/Fruit-Images-Dataset
- **Contents:** 130+ fruit varieties including pears
- **Pros:** ✅ Has pears!, ✅ Multiple apple varieties, ✅ European fruits
- **Cons:** ❌ Only fresh fruits, no ripeness stages
- **Decision:** ⚠️ PARTIAL USE - Only for fresh pear class

### 3.3 Gap Analysis: Missing Data

| Fruit | Fresh | Unripe | Rotten | Action Needed |
|-------|-------|--------|--------|---------------|
| Apple | ✅ | ✅ | ✅ | Complete |
| Banana | ✅ | ✅ | ✅ | Complete |
| Orange | ✅ | ✅ | ✅ | Complete |
| Pear | ✅ | ❌ | ❌ | **Collect own data** |
| Strawberry | ✅ | ❌ | ✅ | **Collect unripe** |
| Grape | ✅ | ❌ | ✅ | **Collect unripe** |

### 3.4 Data Collection Plan for Gaps

For missing data (pears, unripe strawberries/grapes), I will:

1. **Collect own images** at Dutch supermarkets (Albert Heijn, Jumbo)
2. **Document provenance**: Date, location, device used
3. **Ensure diversity**: Different lighting, angles, backgrounds
4. **Label verification**: Have second person verify labels

---

## 4. Data Quality Assessment

### 4.1 Quality Dimensions Analysis

| Dimension | Current Status | Evidence | Action |
|-----------|----------------|----------|--------|
| **Validity** | ✅ Good | Images match fruit ripeness categories | None |
| **Accuracy** | ⚠️ Medium | Some mislabeled images found in review | Manual review of 10% sample |
| **Completeness** | ⚠️ Gaps | Missing pear ripeness, unripe berries | Collect own data |
| **Consistency** | ⚠️ Variable | Different image sizes, backgrounds | Preprocessing pipeline |
| **Uniqueness** | ✅ Good | No duplicates found | Deduplication check |
| **Timeliness** | ✅ Good | Datasets from 2020-2024 | Acceptable |

### 4.2 Bias Assessment

| Potential Bias | Risk | Mitigation |
|----------------|------|------------|
| **Geographic bias** | Datasets mostly from Asia/Americas | Add European fruit photos |
| **Class imbalance** | Some classes have more images | Undersample/oversample |
| **Lighting bias** | Many studio photos | Add real-world photos |
| **Background bias** | White backgrounds dominate | Data augmentation |

---

## 5. Data Preparation Plan

### 5.1 Preprocessing Steps

```
1. Resize all images to 224x224 (MobileNetV2 requirement)
2. Normalize pixel values to [0,1]
3. Remove corrupted/unreadable images
4. Verify and fix incorrect labels
5. Remove duplicates
6. Balance classes (target: 1500 images per class)
```

### 5.2 Data Augmentation Strategy

To improve model robustness and handle class imbalance:

| Augmentation | Reason |
|--------------|--------|
| Horizontal flip | Fruit orientation varies |
| Rotation (±20°) | Phone camera angles vary |
| Brightness (±20%) | Indoor/outdoor lighting |
| Zoom (0.8-1.2x) | Distance to fruit varies |
| Random crop | Focus on different parts |

### 5.3 Train/Validation/Test Split

| Split | Percentage | Purpose |
|-------|------------|---------|
| Training | 70% | Model learning |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation (never seen) |

**Important:** Split is stratified to maintain class distribution.

---

## 6. Ethical & Legal Considerations (GDPR/AI Act)

### 6.1 Data Privacy
- ✅ All datasets are publicly available
- ✅ No personal data in images
- ✅ No faces or identifiable information
- ✅ Own collected data: only fruit images, no store branding

### 6.2 AI Act Classification
- **Risk Level:** Minimal Risk
- **Reason:** Consumer application, no health/safety decisions
- **Obligations:** Transparency about AI use in app

### 6.3 Licensing
| Dataset | License | Commercial Use |
|---------|---------|----------------|
| Kaggle Fruits Fresh/Rotten | CC0 Public Domain | ✅ Allowed |
| Mendeley Dataset | CC BY 4.0 | ✅ With attribution |
| Fruits-360 | MIT License | ✅ Allowed |
| Own collected data | Own work | ✅ Full rights |

---

## 7. Success Criteria

### 7.1 Model Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Validation Accuracy | >95% | 97.3% ✅ |
| Per-class Precision | >90% | TBD |
| Per-class Recall | >90% | TBD |
| Confusion between Fresh/Rotten | <5% | TBD |

### 7.2 Data Quality Targets

| Metric | Target |
|--------|--------|
| Minimum images per class | 1500 |
| Maximum class imbalance ratio | 1:2 |
| Label accuracy (verified) | >98% |

---

## 8. References

1. Kaggle. "Fruits Fresh and Rotten for Classification." https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

2. Kaggle. "Fruit Ripeness: Unripe, Ripe, and Rotten." https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten

3. Mendeley Data. "Fresh and Rotten Fruits Dataset." https://data.mendeley.com/datasets/bdd69gyhv8/1

4. GitHub. "Fruits-360 Dataset." https://github.com/Horea94/Fruit-Images-Dataset

5. European Commission. "Food Waste Statistics." https://ec.europa.eu/food/safety/food-waste

---

## 9. Next Steps

- [ ] Download and merge selected datasets
- [ ] Perform EDA on combined dataset
- [ ] Identify and fix quality issues
- [ ] Collect missing European fruit data
- [ ] Balance all classes
- [ ] Retrain model with improved data
- [ ] Evaluate with confusion matrix per class

---

*Document created following IBM Data Science Methodology and DOT Framework principles.*
