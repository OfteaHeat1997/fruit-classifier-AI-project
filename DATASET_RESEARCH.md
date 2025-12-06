# Dataset Research & Selection Process

**Project:** FreshScan - Fruit Ripeness Classifier
**Author:** Maria Paula Salazar Agudelo
**Date:** December 2025
**MMM Label:** 🤖 AI-Assisted

---

## 1. Research Objective

**Goal:** Find a dataset for training an AI model that can:
- Recognize **many types of fruits** (not just 3)
- Classify **ripeness stages** (Unripe, Ripe/Fresh, Rotten)
- Have **proper academic provenance** (DOI, peer-reviewed)

**Target Users:** Dutch consumers who want to reduce food waste

---

## 2. Requirements Defined

Based on my project context and Fontys curriculum (LO3 - Data Understanding), I defined these requirements:

| Requirement | Priority | Reason |
|-------------|----------|--------|
| Multiple fruit types (10+) | Must Have | App should work with various fruits |
| Ripeness stages (Unripe/Ripe/Rotten) | Must Have | Core functionality of the app |
| Large dataset (10,000+ images) | Should Have | Better model accuracy |
| Academic provenance (DOI) | Must Have | Fontys requirement - know data source |
| Fruits only (no vegetables) | Must Have | App is for fruits, not vegetables |
| European/Dutch relevant fruits | Should Have | Target market is Netherlands |

---

## 3. Datasets Researched

### 3.1 Academic Datasets (With DOI)

#### Dataset A: Mendeley Fresh & Rotten Fruits ✅ SELECTED
| Attribute | Details |
|-----------|---------|
| **Source** | Mendeley Data |
| **DOI** | 10.17632/bdd69gyhv8.1 |
| **Authors** | Sultana, Jahan, Uddin (Jahangirnagar University) |
| **Fruits** | 8 types: Apple, Banana, Orange, Grape, Strawberry, Pomegranate, Guava, Jujube |
| **Stages** | 2: Fresh, Rotten |
| **Images** | 12,335 (augmented) / 3,200 (original) |
| **License** | CC BY 4.0 |
| **Peer Reviewed** | ✅ Yes - Published in Data in Brief journal |

**Pros:**
- ✅ Academic source with DOI
- ✅ Peer-reviewed publication
- ✅ Domain expert verified labels (Ministry of Agriculture)
- ✅ Clear collection methodology documented
- ✅ Good variety of fruits (8 types)
- ✅ Includes European fruits (Apple, Orange, Grape, Strawberry)

**Cons:**
- ❌ Only 2 ripeness stages (no "Unripe" category)
- ❌ Some tropical fruits not common in Netherlands (Guava, Jujube)

---

#### Dataset B: Fruits-360 ⚠️ PARTIALLY USEFUL
| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle / GitHub |
| **DOI** | 10.2478/ausi-2018-0002 |
| **Authors** | Muresan, Oltean (Babeș-Bolyai University, Romania) |
| **Fruits** | 131 types (includes vegetables) |
| **Stages** | 1: Fresh only |
| **Images** | 90,483 |
| **License** | MIT |

**Pros:**
- ✅ Academic source with DOI
- ✅ European origin (Romania)
- ✅ Many fruit varieties including Pear
- ✅ Large dataset

**Cons:**
- ❌ Mixed with vegetables - need to filter
- ❌ NO ripeness stages (only fresh fruits)
- ❌ Cannot classify Unripe or Rotten

**Conclusion:** Not suitable for ripeness classification, only fruit identification.

---

### 3.2 Kaggle Datasets (No DOI)

#### Dataset C: Fruit Ripeness - Unripe, Ripe, Rotten
| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle |
| **URL** | kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten |
| **DOI** | ❌ None |
| **Fruits** | 3 types: Apple, Banana, Orange |
| **Stages** | 3: Unripe, Ripe, Rotten |
| **Size** | 3.9 GB |
| **License** | CC BY-SA 4.0 |

**Pros:**
- ✅ Has ALL 3 ripeness stages (Unripe, Ripe, Rotten)
- ✅ Large dataset
- ✅ Clear license

**Cons:**
- ❌ No DOI - cannot verify source
- ❌ Unknown collection method
- ❌ Only 3 fruit types
- ❌ Hard to defend for academic portfolio

---

#### Dataset D: Fruits Ripeness Classification (5 fruits)
| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle |
| **URL** | kaggle.com/datasets/asadullahprl/fruits-ripeness-classification-dataset |
| **DOI** | ❌ None |
| **Fruits** | 5 types: Apple, Banana, Mango, Orange, Tomato |
| **Stages** | 3: Unripe, Ripe, Overripe |
| **Size** | 183 MB |
| **License** | CC0 Public Domain |

**Pros:**
- ✅ Has 3 ripeness stages
- ✅ 5 fruit types
- ✅ Public domain license

**Cons:**
- ❌ No DOI
- ❌ Includes Tomato (is it fruit or vegetable?)
- ❌ Small dataset
- ❌ Unknown provenance

---

#### Dataset E: Fruit Image Dataset 22 Classes
| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle |
| **URL** | kaggle.com/datasets/mdsagorahmed/fruit-image-dataset-22-classes |
| **DOI** | ❌ None |
| **Fruits** | 11 types |
| **Stages** | 2: Ripe, Unripe |
| **Images** | 8,700+ |
| **Size** | 41 MB |

**Pros:**
- ✅ 11 different fruits
- ✅ Has Unripe stage
- ✅ Fruits only (no vegetables)

**Cons:**
- ❌ No DOI
- ❌ No "Rotten" category
- ❌ Small file size (lower quality?)
- ❌ Unknown source

---

#### Dataset F: Fruit & Vegetable Disease (Healthy vs Rotten)
| Attribute | Details |
|-----------|---------|
| **Source** | Kaggle |
| **URL** | kaggle.com/datasets/muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten |
| **DOI** | ❌ None |
| **Types** | 14 (mixed fruits and vegetables) |
| **Stages** | 2: Healthy, Rotten |
| **Classes** | 28 |

**Pros:**
- ✅ Many classes
- ✅ High quality images

**Cons:**
- ❌ No DOI
- ❌ Mixed with vegetables
- ❌ No "Unripe" category

---

### 3.3 Other Sources Explored

#### Roboflow - Fruitectives Team
| Attribute | Details |
|-----------|---------|
| **Source** | Roboflow Universe |
| **Fruits** | 8 types: Apple, Banana, Grape, Mango, Melon, Orange, Peach, Pear |
| **Stages** | 4: Unripe, Ripe, Overripe, Rotten |
| **Classes** | 32 |

**Pros:**
- ✅ 4 ripeness stages (best variety!)
- ✅ 8 fruit types
- ✅ Includes Pear (European)
- ✅ CC BY 4.0 license

**Cons:**
- ❌ No DOI
- ❌ User-uploaded (unknown quality)
- ❌ No peer review
- ❌ Cannot verify label accuracy

---

#### GitHub - Fruit Ripeness Detection (anujgoenka9)
| Attribute | Details |
|-----------|---------|
| **Fruits** | 6 types: Apple, Banana, Orange, Pomegranate, Mango, Papaya |
| **Stages** | 3: Raw, Ripe, Rotten |
| **Images** | ~1,980 |

**Pros:**
- ✅ 3 ripeness stages
- ✅ 6 fruit types

**Cons:**
- ❌ No DOI
- ❌ Images scraped from Google (copyright issues!)
- ❌ Small dataset
- ❌ Not downloadable directly

---

## 4. Comparison Matrix

| Dataset | Fruits | Stages | Images | DOI | Fruits Only | Score |
|---------|--------|--------|--------|-----|-------------|-------|
| **Mendeley** | 8 | 2 | 12,335 | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| Fruits-360 | 131 | 1 | 90,483 | ✅ | ❌ | ⭐⭐⭐ |
| Kaggle Ripeness 3.9GB | 3 | 3 | Large | ❌ | ✅ | ⭐⭐⭐ |
| Kaggle 5 Fruits | 5 | 3 | ~2,000 | ❌ | ⚠️ | ⭐⭐ |
| Kaggle 22 Classes | 11 | 2 | 8,700 | ❌ | ✅ | ⭐⭐⭐ |
| Roboflow | 8 | 4 | 6,309 | ❌ | ✅ | ⭐⭐⭐ |

**Scoring Criteria:**
- ⭐⭐⭐⭐⭐ = Has DOI + Good variety + Proper provenance
- ⭐⭐⭐⭐ = Has DOI OR very large with good variety
- ⭐⭐⭐ = Useful but missing DOI or limited stages
- ⭐⭐ = Limited usefulness
- ⭐ = Not recommended

---

## 5. Key Learnings

### 5.1 What I Discovered

1. **No "perfect" dataset exists** that has:
   - Many fruit types (20+)
   - All ripeness stages (Unripe, Ripe, Rotten)
   - Academic provenance (DOI)
   - Large size (50,000+ images)

2. **Trade-offs are necessary:**
   - Academic datasets (DOI) have fewer fruits/stages
   - Kaggle datasets have more variety but unknown sources
   - Roboflow has best stages but no academic verification

3. **Provenance matters for portfolio:**
   - Fontys requires knowing WHERE data comes from
   - DOI allows verification and citation
   - "Unknown source" is not acceptable for academic work

4. **Combining datasets is an option** but adds complexity:
   - Different image qualities
   - Different labeling standards
   - Need to document each source

### 5.2 Research Gap Identified

There is a **gap in available datasets** for fruit ripeness classification:
- Most academic datasets focus on Fresh vs Rotten (2 stages)
- "Unripe" stage is rarely included in academic datasets
- Large multi-fruit ripeness datasets lack academic provenance

**This could be future research opportunity!**

---

## 6. Final Decision

### Selected: Mendeley Dataset (DOI: 10.17632/bdd69gyhv8.1)

**Reasons:**

| Criteria | Why Mendeley |
|----------|--------------|
| **Academic Requirement** | Has DOI, peer-reviewed paper |
| **Provenance** | Authors, institution, methodology documented |
| **Verification** | Domain expert from Ministry of Agriculture verified labels |
| **Size** | 12,335 images is sufficient for transfer learning |
| **Variety** | 8 fruits × 2 stages = 16 classes |
| **License** | CC BY 4.0 allows use with attribution |
| **Defensibility** | Can explain and cite in portfolio presentation |

### What I'm Giving Up:

| Missing Feature | Mitigation |
|-----------------|------------|
| No "Unripe" stage | Focus on Fresh vs Rotten (still useful for users) |
| Only 8 fruits | Cover most common fruits in Netherlands |
| Some tropical fruits | Can filter to only European-relevant fruits |

---

## 7. Justification for Portfolio (LO3)

### How This Meets Fontys Requirements:

| LO3 Requirement | How I Met It |
|-----------------|--------------|
| **Data provenance documented** | DOI: 10.17632/bdd69gyhv8.1 |
| **Know where data comes from** | Academic paper, known authors |
| **Collection method known** | Documented in peer-reviewed publication |
| **Quality verified** | Domain expert verification |
| **Bias considered** | Documented geographic bias (Bangladesh origin) |
| **Research process documented** | This document! |

### Citation:
```
Sultana N, Jahan M, Uddin MS.
"An extensive dataset for successful recognition of fresh and rotten fruits."
Data in Brief. 2022 Aug 24;44:108552.
DOI: 10.1016/j.dib.2022.108552
```

---

## 8. Alternative Options Considered

If I had more time or different requirements, I could:

### Option A: Combine Academic + Kaggle
- Use Mendeley as primary (cite DOI)
- Add Kaggle data as "supplementary" (acknowledge limitation)
- Document clearly which images come from where

### Option B: Collect Own Data
- Photograph fruits at Dutch supermarkets (Albert Heijn, Jumbo)
- Full control over provenance
- Time-consuming but 100% traceable

### Option C: Use Roboflow for More Stages
- Accept no DOI limitation
- Document clearly in portfolio
- Acknowledge this is a limitation

---

## 9. Conclusion

**For my portfolio project, academic provenance is more important than dataset size.**

A smaller dataset with proper DOI that I can explain and defend is better than a large dataset from unknown sources.

### Final Dataset Configuration:

| Attribute | Value |
|-----------|-------|
| **Dataset** | Mendeley Fresh & Rotten Fruits |
| **DOI** | 10.17632/bdd69gyhv8.1 |
| **Fruits** | 8 (Apple, Banana, Orange, Grape, Strawberry, Pomegranate, Guava, Jujube) |
| **Stages** | 2 (Fresh, Rotten) |
| **Classes** | 16 |
| **Images** | 12,335 |
| **License** | CC BY 4.0 |

---

## 10. References

1. Sultana N, Jahan M, Uddin MS. "An extensive dataset for successful recognition of fresh and rotten fruits." Data in Brief. 2022. DOI: 10.1016/j.dib.2022.108552

2. Muresan H, Oltean M. "Fruit recognition from images using deep learning." Acta Univ. Sapientiae. 2018. DOI: 10.2478/ausi-2018-0002

3. Kaggle. "Fruit Ripeness: Unripe, Ripe, and Rotten." https://www.kaggle.com/datasets/leftin/fruit-ripeness-unripe-ripe-and-rotten

4. Kaggle. "Fruit Image Dataset: 22 Classes." https://www.kaggle.com/datasets/mdsagorahmed/fruit-image-dataset-22-classes

5. Roboflow. "Fruit Ripeness by Fruitectives Team." https://universe.roboflow.com/fruitectives-team/fruit-ripeness-unjex

---

*This document demonstrates my research process following the DOT Framework (Literature Study) and IBM Data Science Methodology (Data Requirements, Data Collection, Data Understanding).*

**Document Created:** December 2025
**Last Updated:** December 2025
