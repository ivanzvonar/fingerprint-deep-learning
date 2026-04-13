# 🏗️ Model Architecture

## Convolutional Neural Network (CNN) Overview

Both models share the same core CNN architecture — the key difference is the size of the Dense layer and training strategy.

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT LAYER                         │
│              Grayscale image 96 × 103 px                │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              CONV BLOCK 1                               │
│   Conv2D(32 filters, 3×3) → ReLU → MaxPooling(2×2)     │
│   Extracts: edges, basic textures                       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              CONV BLOCK 2                               │
│   Conv2D(64 filters, 3×3) → ReLU → MaxPooling(2×2)     │
│   Extracts: ridge patterns, local structures            │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              CONV BLOCK 3                               │
│   Conv2D(128 filters, 3×3) → ReLU → MaxPooling(2×2)    │
│   Extracts: minutiae, bifurcations, complex patterns    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    FLATTEN                              │
│         Converts 3D feature maps → 1D vector           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              FULLY CONNECTED                            │
│   Dense(128)* → ReLU → Dropout(0.5)                    │
│   *Model 2 uses Dense(256)                             │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  OUTPUT LAYER                           │
│         Dense(2) → Softmax                             │
│         Classes: [Real=0, Altered=1]                   │
└─────────────────────────────────────────────────────────┘
```

---

## Layer-by-Layer Explanation

### Convolutional Layers (Conv2D)

Each convolutional layer applies a set of learnable filters across the input image. For fingerprint analysis:

- **32 filters** in the first layer capture basic features like edges and ridge direction
- **64 filters** in the second layer combine those into more complex ridge patterns
- **128 filters** in the third layer detect high-level features like minutiae points, bifurcations, and overall print topology

### MaxPooling Layers

After each Conv2D, a MaxPooling(2×2) layer:
- Reduces the spatial dimensions by half
- Keeps only the strongest activations (most prominent features)
- Reduces computation and helps prevent overfitting

### Flatten Layer

Converts the 3D tensor output of the last pooling layer into a 1D vector so it can be fed into fully-connected layers.

### Dense + Dropout

- `Dense(128)` / `Dense(256)` — learns combinations of extracted features
- `Dropout(0.5)` — randomly disables 50% of neurons during training to prevent overfitting

### Output Layer

- `Dense(2, activation='softmax')` — outputs two probability values that sum to 1.0
- The class with the higher probability is the prediction: **Real** or **Altered**

---

## Key Differences: Model 1 vs Model 2

| Aspect | Model 1 | Model 2 |
|---|---|---|
| Training data | All fingers combined | Separate per finger |
| Dense layer | 128 neurons | 256 neurons |
| Epochs | 20 | Up to 50 |
| Early stopping | No | Yes (patience=5) |
| Augmentation rotation | ±20° | ±30° |
| Augmentation shifts | 20% | 30% |
| Output models | 1 | 5 (one per finger) |

---

## Activation Functions

| Function | Used in | Purpose |
|---|---|---|
| **ReLU** | Conv2D, Dense hidden layers | Introduces non-linearity, prevents vanishing gradients |
| **Softmax** | Output layer | Converts raw scores to class probabilities |

---

## Loss Function & Optimizer

- **Loss:** `sparse_categorical_crossentropy` — standard for integer-labeled multi-class classification
- **Optimizer:** `Adam` — adaptive learning rate, well-suited for image classification tasks
- **Metric:** `accuracy`

---

## Data Augmentation

Augmentation generates artificial variations of training images to improve model generalization:

| Transformation | Model 1 | Model 2 |
|---|---|---|
| Rotation | ±20° | ±30° |
| Width shift | 20% | 30% |
| Height shift | 20% | 30% |
| Shear | 20% | 30% |
| Zoom | 20% | 30% |
| Horizontal flip | Yes | Yes |
| Fill mode | nearest | nearest |

This is particularly important for latent fingerprint analysis where prints can appear at different orientations and positions on surfaces.
