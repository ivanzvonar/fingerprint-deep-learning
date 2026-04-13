"""
Model 1 - General CNN Fingerprint Classifier
=============================================
Classifies fingerprints as "Real" or "Altered" regardless of finger type.

Dataset: SOCOFing (Sokoto Coventry Fingerprint Dataset)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ──────────────────────────────────────────────
# Parameters
# ──────────────────────────────────────────────
IMG_SIZE = (96, 103)
BATCH_SIZE = 32
DATASET_PATH = "Dataset"


# ──────────────────────────────────────────────
# Load Dataset
# ──────────────────────────────────────────────
def load_dataset(dataset_path):
    """
    Loads fingerprint images from the Dataset directory.
    Labels: 0 = Real, 1 = Altered
    Returns normalized numpy arrays.
    """
    images = []
    labels = []
    classes = {"Real": 0, "Altered": 1}

    for class_label in classes:
        class_path = os.path.join(dataset_path, class_label)
        if os.path.exists(class_path):
            for root, _, files in os.walk(class_path):
                for filename in files:
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
                    except PermissionError as e:
                        print(f"Skipping {img_path}: {e}")

    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)
    return images, labels


# ──────────────────────────────────────────────
# Build Model
# ──────────────────────────────────────────────
def build_model():
    """
    CNN architecture:
      3x Conv2D + MaxPooling blocks for feature extraction
      Dense(128) + Dropout(0.5) for classification
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
        layers.Dense(128, activation='relu'),
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
def compute_metrics(y_true, y_scores):
    """
    Computes biometric evaluation metrics:
      - EER  (Equal Error Rate)
      - GAR  (Genuine Acceptance Rate)
      - FAR  (False Acceptance Rate)
      - FRR  (False Rejection Rate)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr  # False Negative Rate (FRR)

    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    gar = 1 - fnr

    far = fpr
    frr = fnr

    print(f"\n── Evaluation Metrics ──────────────────────")
    print(f"Equal Error Rate  (EER): {eer:.4f}")
    print(f"Genuine Accept.   (GAR): {np.max(gar):.4f}")
    print(f"False Accept.     (FAR): {far[np.nanargmin(np.absolute(fnr - fpr))]:.4f}")
    print(f"False Rejection   (FRR): {frr[np.nanargmin(np.absolute(fnr - fpr))]:.4f}")

    return fpr, tpr, eer, gar, far, frr


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data
    print("Loading dataset...")
    images, labels = load_dataset(DATASET_PATH)
    print(f"Loaded {len(images)} images.")

    # 2. Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    # 3. Data augmentation
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # 4. Build and train model
    model = build_model()
    model.summary()

    print("\nTraining Model 1...")
    history = model.fit(
        data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_test, y_test),
        epochs=20,
        verbose=1
    )

    # 5. Save model
    model.save("fingerprint_recognition_model.h5")
    print("Model saved to fingerprint_recognition_model.h5")

    # 6. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest Accuracy: {accuracy:.2f}")

    y_scores = model.predict(X_test)[:, 1]
    compute_metrics(y_test, y_scores)

    # 7. Plot training history
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model 1 — Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model 1 — Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/model1_training.png')
    plt.show()
