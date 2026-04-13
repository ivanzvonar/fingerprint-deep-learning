"""
Model 2 - Per-Finger CNN Classifier
=====================================
Trains a separate CNN model for each finger (thumb, index, middle, ring, little).
Includes stronger augmentation and early stopping.

Dataset: SOCOFing (Sokoto Coventry Fingerprint Dataset)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
IMG_SIZE = (96, 103)
BATCH_SIZE = 32
DATASET_PATH = "Dataset"

FINGER_NAMES = ["thumb", "index", "middle", "ring", "little"]


# ──────────────────────────────────────────────
# Load Dataset for a Specific Finger
# ──────────────────────────────────────────────
def load_finger_dataset(dataset_path, finger_name):
    """
    Loads images only for the specified finger.
    Filters filenames by finger_name string.
    Labels: 0 = Real, 1 = Altered
    """
    images, labels = [], []
    classes = {"Real": 0, "Altered": 1}

    for class_label in classes:
        class_path = os.path.join(dataset_path, class_label)
        if os.path.exists(class_path):
            for root, _, files in os.walk(class_path):
                for filename in files:
                    if finger_name in filename:
                        img_path = os.path.join(root, filename)
                        try:
                            image = keras.preprocessing.image.load_img(
                                img_path,
                                target_size=IMG_SIZE,
                                color_mode='grayscale'
                            )
                            image = keras.preprocessing.image.img_to_array(image)
                            images.append(image)
                            labels.append(classes[class_label])
                        except Exception as e:
                            print(f"Skipping {img_path}: {e}")

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels


# ──────────────────────────────────────────────
# Build Model
# ──────────────────────────────────────────────
def build_model():
    """
    CNN architecture — same as Model 1 but with Dense(256) instead of Dense(128).

    3x Conv2D + MaxPooling blocks for feature extraction
    Dense(256) + Dropout(0.5) — larger dense layer for per-finger specialization
    Softmax output for 2 classes (Real / Altered)
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),   # increased from 128 → 256
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ──────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────
def evaluate_metrics(model, X_test, y_test, finger_name):
    """
    Computes and prints biometric evaluation metrics for a given finger model.
      - EER  (Equal Error Rate)
      - GAR  (Genuine Acceptance Rate)
      - FAR  (False Acceptance Rate)
      - FRR  (False Rejection Rate)
    """
    y_scores = model.predict(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    eer_index = np.argmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_index]
    gar = tpr[eer_index]
    far = fpr[eer_index]
    frr = 1 - gar

    print(f"\n── {finger_name.upper()} Metrics ──────────────────────")
    print(f"EER : {eer:.4f}")
    print(f"GAR : {gar:.4f}")
    print(f"FAR : {far:.4f}")
    print(f"FRR : {frr:.4f}")

    return {"finger": finger_name, "eer": eer, "gar": gar, "far": far, "frr": frr}


# ──────────────────────────────────────────────
# Main — Train one model per finger
# ──────────────────────────────────────────────
if __name__ == "__main__":

    all_results = []

    for finger in FINGER_NAMES:
        print(f"\n{'='*50}")
        print(f"Training model for: {finger.upper()}")
        print('='*50)

        # 1. Load finger-specific data
        images, labels = load_finger_dataset(DATASET_PATH, finger)
        print(f"Loaded {len(images)} images for {finger}.")

        # 2. Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

        # 3. Data augmentation (stronger than Model 1)
        data_gen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # 4. Build model
        model = build_model()

        # 5. Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # 6. Train
        history = model.fit(
            data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_test, y_test),
            epochs=50,
            callbacks=[early_stopping],
            verbose=1
        )

        # 7. Save model
        model_path = f"fingerprint_model_{finger}.h5"
        model.save(model_path)
        print(f"Model saved to {model_path}")

        # 8. Evaluate
        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f"Test Accuracy for {finger}: {accuracy:.2f}")

        result = evaluate_metrics(model, X_test, y_test, finger)
        result["accuracy"] = accuracy
        all_results.append(result)

    # ──────────────────────────────────────────────
    # Summary Table
    # ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Finger':<10} {'Accuracy':>10} {'EER':>8} {'GAR':>8} {'FAR':>8} {'FRR':>8}")
    print("-"*60)
    for r in all_results:
        print(
            f"{r['finger']:<10} "
            f"{r['accuracy']:>10.2%} "
            f"{r['eer']:>8.4f} "
            f"{r['gar']:>8.4f} "
            f"{r['far']:>8.4f} "
            f"{r['frr']:>8.4f}"
        )
