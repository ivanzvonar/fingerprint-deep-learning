# Latent Fingerprint Analysis Using Deep Learning

---

## Overview

This project investigates the application of **deep learning techniques** for analyzing **latent fingerprints** using Convolutional Neural Networks (CNN). Latent fingerprints are often of poor quality, which makes identification with classical methods difficult.

Two CNN models were developed:
- **Model 1** — General classification of fingerprints (Real vs. Altered)
- **Model 2** — Per-finger classification with individual models for each of the 5 fingers

---

## Project Structure

```
fingerprint-deep-learning/
│
├── src/
│   ├── model1.py          # General CNN classifier (Real vs Altered)
│   └── model2.py          # Per-finger CNN classifiers
│
├── results/
│   └── README.md          # Evaluation results and metrics summary
│
├── docs/
│   └── architecture.md    # Detailed model architecture documentation
│
└── README.md
```

---

## Dataset

The project uses the **Sokoto Coventry Fingerprint Dataset (SOCOFing)**:

| Property | Details |
|---|---|
| Total fingerprints | 6,000 originals + 17,934 synthetic |
| Subjects | 600 (all 18+) |
| Fingers per subject | 10 (both hands) |
| Image format | Grayscale BMP |
| Resolution | 96 × 103 pixels |
| Labels | Gender (M/F), Hand (L/R), Finger name |

**Synthetic alterations** (3 difficulty levels: easy, medium, hard):
- `Obl` — Obliteration (partial/full ridge erasure)
- `CR` — Central Rotation
- `Z-cut` — Segmentation

**File naming convention:**
```
001_M_Left_little_finger_Obl.bmp
 ^   ^   ^      ^        ^
 ID  Sex Hand  Finger  Alteration
```

---

## Models

### Model 1 — General Classifier

A standard CNN that classifies **all fingerprints** regardless of finger type.

**Architecture:**
```
Input (96×103×1)
→ Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)
→ Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)
→ Conv2D(128, 3×3, ReLU) → MaxPooling(2×2)
→ Flatten
→ Dense(128, ReLU) → Dropout(0.5)
→ Dense(2, Softmax)
```

| Parameter | Value |
|---|---|
| Input size | 96 × 103 px (grayscale) |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Train/Test split | 80% / 20% |

### Model 2 — Per-Finger Classifier

Trains a **separate CNN model for each finger** with stronger augmentation and early stopping.

**Key differences from Model 1:**
- Dense layer increased from 128 → 256 neurons
- Augmentation parameters increased (rotation up to 30°, shifts up to 30%)
- Early stopping with `patience=5` on `val_loss`
- Up to 50 training epochs per finger

---

## Results

### Model 1 — Overall Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | **95%** |
| Equal Error Rate (EER) | 6.22% |
| Genuine Acceptance Rate (GAR) | 100% |
| False Acceptance Rate (FAR) | 6.22% |
| False Rejection Rate (FRR) | 6.24% |

### Model 2 — Per-Finger Performance

| Finger | Accuracy | EER | GAR | FAR | FRR |
|---|---|---|---|---|---|
| Thumb | 89% | 17.13% | 82.68% | 17.13% | 17.32% |
| Index | 88% | 15.38% | 84.71% | 15.38% | 15.29% |
| Middle | 90% | 15.32% | 84.59% | 15.32% | 15.41% |
| Ring | 90% | 14.46% | 85.54% | 14.46% | 14.46% |
| Little | 88% | 17.74% | 82.36% | 17.74% | 17.64% |

**Key observations:**
- Model 1 outperforms Model 2 on overall accuracy (95% vs ~88–90%)
- Ring finger achieved the best per-finger results (lowest EER: 14.46%)
- Little finger was the most challenging (highest EER: 17.74%)

---

## Setup & Usage

### Requirements

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

### Dataset Setup

1. Download [SOCOFing dataset](https://www.kaggle.com/datasets/ruizgara/socofing)
2. Organize into the following structure:

```
Dataset/
├── Real/
│   └── *.bmp
└── Altered/
    ├── Altered-Easy/
    ├── Altered-Medium/
    └── Altered-Hard/
```

### Run Model 1

```bash
python src/model1.py
```

### Run Model 2

```bash
python src/model2.py
```

---

## Key Concepts

| Term | Description |
|---|---|
| **CNN** | Convolutional Neural Network — extracts spatial features from images |
| **Latent fingerprint** | An unintentional fingerprint left on a surface, often invisible to the naked eye |
| **EER** | Equal Error Rate — point where FAR = FRR; lower is better |
| **GAR** | Genuine Acceptance Rate — how often real fingerprints are correctly accepted |
| **FAR** | False Acceptance Rate — how often fake prints are wrongly accepted |
| **FRR** | False Rejection Rate — how often real prints are wrongly rejected |
| **Augmentation** | Generating image variations (rotations, shifts) to improve model robustness |
| **Early Stopping** | Stops training when validation loss stops improving |

---
