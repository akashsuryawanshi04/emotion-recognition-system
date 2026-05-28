"""

Author - Akash Suryawanshi 
train/train_speech.py
Full training pipeline for Speech Emotion Recognition.

Dataset  : RAVDESS
           Download from: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
           Extract so that: data/ravdess/ contains Actor_01/ ... Actor_24/ folders

Filename format : 03-01-05-01-02-01-12.wav
    Digit 3 (index 2) = Emotion:
        01=neutral, 02=calm, 03=happy, 04=sad,
        05=angry, 06=fearful, 07=disgust, 08=surprised

Run:
    python train/train_speech.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.speech_model import build_speech_model, SPEECH_EMOTIONS, N_MFCC, MAX_FRAMES
from utils.audio_utils import extract_mfcc

# ── Config ──────────────────────────────────────────────
DATASET_PATH = os.path.join('data', 'ravdess')
MODEL_SAVE   = os.path.join('saved_models', 'speech_emotion_model.h5')
BATCH_SIZE   = 32
EPOCHS       = 80
TEST_SIZE    = 0.2
os.makedirs('saved_models', exist_ok=True)

# RAVDESS emotion code → label mapping
# We drop 'calm' (02) and map to our 7 standard emotions
RAVDESS_EMOTION_MAP = {
    '01': 'Neutral',
    '02': None,        # calm — skip
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fear',
    '07': 'Disgust',
    '08': 'Surprise'
}


def load_ravdess_data():
    """
    Load all RAVDESS audio files, extract MFCC features, return X and y arrays.
    """
    print("\n[1/5] Loading RAVDESS dataset and extracting MFCC features...")

    X, y = [], []
    audio_files = glob.glob(os.path.join(DATASET_PATH, 'Actor_*', '*.wav'))

    if len(audio_files) == 0:
        raise FileNotFoundError(
            f"No audio files found in {DATASET_PATH}.\n"
            "Download RAVDESS from:\n"
            "https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio\n"
            "Extract so: data/ravdess/Actor_01/ ... Actor_24/ folders exist."
        )

    print(f"    Found {len(audio_files)} audio files")

    for i, filepath in enumerate(audio_files):
        filename = os.path.basename(filepath)
        parts = filename.replace('.wav', '').split('-')

        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        emotion_label = RAVDESS_EMOTION_MAP.get(emotion_code)

        if emotion_label is None:
            continue  # skip 'calm'

        try:
            mfcc = extract_mfcc(audio_path=filepath)
            X.append(mfcc)
            y.append(emotion_label)
        except Exception as e:
            print(f"    Warning: Could not process {filename}: {e}")
            continue

        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(audio_files)} files...")

    X = np.array(X)                     # (N, N_MFCC, MAX_FRAMES)
    X = X[:, :, :, np.newaxis]          # (N, N_MFCC, MAX_FRAMES, 1)
    print(f"\n    Total samples : {len(X)}")
    print(f"    Feature shape : {X.shape[1:]}")

    # Encode labels
    le = LabelEncoder()
    le.fit(SPEECH_EMOTIONS)
    y_encoded = le.transform(y)
    y_onehot = to_categorical(y_encoded, num_classes=len(SPEECH_EMOTIONS))

    print(f"    Label distribution:")
    for emo in SPEECH_EMOTIONS:
        count = sum(1 for label in y if label == emo)
        print(f"      {emo:<12}: {count}")

    return X, y_onehot, y


def train(model, X_train, y_train, X_val, y_val):
    print("\n[2/5] Training model...")
    callbacks = [
        ModelCheckpoint(MODEL_SAVE, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history


def evaluate(model, X_test, y_test_raw):
    print("\n[3/5] Evaluating on test set...")
    preds = model.predict(X_test, verbose=0)
    y_pred_idx = np.argmax(preds, axis=1)
    y_true_idx = np.array([SPEECH_EMOTIONS.index(l) for l in y_test_raw])

    acc = np.mean(y_pred_idx == y_true_idx)
    print(f"\n    Test Accuracy : {acc * 100:.2f}%")
    print("\n--- Classification Report ---")
    print(classification_report(y_true_idx, y_pred_idx, target_names=SPEECH_EMOTIONS))
    return acc, y_true_idx, y_pred_idx


def plot_confusion_matrix(y_true, y_pred):
    print("[4/5] Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=SPEECH_EMOTIONS, yticklabels=SPEECH_EMOTIONS)
    plt.title('Speech Emotion Model — Confusion Matrix', fontsize=14)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('saved_models/speech_confusion_matrix.png', dpi=150)
    plt.show()
    print("    Saved: saved_models/speech_confusion_matrix.png")


def plot_history(history):
    print("[5/5] Plotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val', linewidth=2)
    axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val', linewidth=2)
    axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle('Speech Emotion Model — Training History', fontsize=13)
    plt.tight_layout()
    plt.savefig('saved_models/speech_training_history.png', dpi=150)
    plt.show()
    print("    Saved: saved_models/speech_training_history.png")


if __name__ == '__main__':
    print("=" * 55)
    print("   SPEECH EMOTION RECOGNITION — TRAINING PIPELINE")
    print("=" * 55)

    X, y_onehot, y_raw = load_ravdess_data()

    X_train, X_test, y_train, y_test_oh, y_train_raw, y_test_raw = \
        train_test_split(X, y_onehot, y_raw, test_size=TEST_SIZE,
                         random_state=42, stratify=y_raw)

    print(f"\n    Train: {len(X_train)} | Test: {len(X_test)}")

    model = build_speech_model()
    model.summary()

    history = train(model, X_train, y_train, X_test, y_test_oh)

    best_model = load_model(MODEL_SAVE)
    acc, y_true_idx, y_pred_idx = evaluate(best_model, X_test, y_test_raw)
    plot_confusion_matrix(y_true_idx, y_pred_idx)
    plot_history(history)

    print("\n" + "=" * 55)
    print(f"   DONE! Accuracy: {acc * 100:.2f}%")
    print(f"   Model saved : {MODEL_SAVE}")
    print("=" * 55)
