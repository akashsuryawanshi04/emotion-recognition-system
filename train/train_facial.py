"""

Author - Akash Suryawanshi 
train/train_facial.py
Full training pipeline for Facial Emotion Recognition.

Dataset  : FER2013
           Download from: https://www.kaggle.com/datasets/msambare/fer2013
           Extract so that: data/fer2013/train/ and data/fer2013/test/ exist

Run:
    python train/train_facial.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.facial_model import build_facial_model, FACIAL_EMOTIONS, IMG_SIZE

# ── Config ──────────────────────────────────────────────
DATASET_PATH   = os.path.join('data', 'fer2013')
MODEL_SAVE     = os.path.join('saved_models', 'facial_emotion_model.h5')
BATCH_SIZE     = 64
EPOCHS         = 50
os.makedirs('saved_models', exist_ok=True)


def load_data():
    print("\n[1/5] Loading FER2013 dataset...")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    print(f"    Train samples : {train_gen.samples}")
    print(f"    Test  samples : {test_gen.samples}")
    print(f"    Classes       : {list(train_gen.class_indices.keys())}")
    return train_gen, test_gen


def train(model, train_gen, test_gen):
    print("\n[2/5] Training model...")
    callbacks = [
        ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=callbacks,
        verbose=1
    )
    return history


def evaluate(model, test_gen):
    print("\n[3/5] Evaluating on test set...")
    loss, acc = model.evaluate(test_gen, verbose=0)
    print(f"\n    Test Loss     : {loss:.4f}")
    print(f"    Test Accuracy : {acc * 100:.2f}%")

    test_gen.reset()
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=FACIAL_EMOTIONS))

    return acc, y_true, y_pred


def plot_confusion_matrix(y_true, y_pred):
    print("[4/5] Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=FACIAL_EMOTIONS, yticklabels=FACIAL_EMOTIONS)
    plt.title('Facial Emotion Model — Confusion Matrix', fontsize=14)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('saved_models/facial_confusion_matrix.png', dpi=150)
    plt.show()
    print("    Saved: saved_models/facial_confusion_matrix.png")


def plot_history(history):
    print("[5/5] Plotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('Facial Emotion Model — Training History', fontsize=13)
    plt.tight_layout()
    plt.savefig('saved_models/facial_training_history.png', dpi=150)
    plt.show()
    print("    Saved: saved_models/facial_training_history.png")


if __name__ == '__main__':
    print("=" * 55)
    print("   FACIAL EMOTION RECOGNITION — TRAINING PIPELINE")
    print("=" * 55)

    # Check dataset
    if not os.path.exists(os.path.join(DATASET_PATH, 'train')):
        print(f"\nERROR: Dataset not found at '{DATASET_PATH}/train'")
        print("Steps:")
        print("  1. Go to https://www.kaggle.com/datasets/msambare/fer2013")
        print("  2. Download and extract the dataset")
        print("  3. Place it so 'data/fer2013/train/' and 'data/fer2013/test/' exist")
        sys.exit(1)

    train_gen, test_gen = load_data()
    model = build_facial_model()
    model.summary()

    history = train(model, train_gen, test_gen)

    print("\nLoading best saved model for evaluation...")
    best_model = load_model(MODEL_SAVE)

    acc, y_true, y_pred = evaluate(best_model, test_gen)
    plot_confusion_matrix(y_true, y_pred)
    plot_history(history)

    print("\n" + "=" * 55)
    print(f"   DONE! Accuracy: {acc * 100:.2f}%")
    print(f"   Model saved : {MODEL_SAVE}")
    print("=" * 55)
