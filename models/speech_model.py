"""
models/speech_model.py
CNN + LSTM model for Speech Emotion Recognition.
Trained on RAVDESS dataset using MFCC features.
"""

import numpy as np
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Dropout, LSTM, Dense, Reshape,
    TimeDistributed, Flatten, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam

SPEECH_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
N_MFCC = 40
MAX_FRAMES = 200
NUM_CLASSES = 7


def build_speech_model():
    """
    CNN + LSTM model for speech emotion recognition.
    Input : MFCC feature array (N_MFCC, MAX_FRAMES, 1) = (40, 200, 1)
    Output: 7-class softmax probability vector

    Architecture:
        CNN layers extract local audio patterns from MFCC
        LSTM captures temporal sequence in speech
        Dense output predicts emotion
    """
    inputs = Input(shape=(N_MFCC, MAX_FRAMES, 1), name='audio_input')

    # CNN block 1
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # CNN block 2
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # CNN block 3
    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 4))(x)
    x = Dropout(0.25)(x)

    # Reshape for LSTM: (batch, time_steps, features)
    shape = x.shape
    time_steps = shape[2]
    features = shape[1] * shape[3]
    x = Reshape((time_steps, features))(x)

    # LSTM layers
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(64)(x)
    x = Dropout(0.3)(x)

    # Output
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name='speech_output')(x)

    model = Model(inputs, outputs, name='CNN_LSTM_Speech')
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_speech_model(path='saved_models/speech_emotion_model.h5'):
    return load_model(path)


def predict_speech_emotion(model, mfcc_features):
    """
    Predict emotion from MFCC feature array.
    mfcc_features: (N_MFCC, MAX_FRAMES) numpy array
    Returns: (emotion_label, confidence, all_scores_dict)
    """
    x = mfcc_features[np.newaxis, :, :, np.newaxis]  # (1, 40, 200, 1)
    scores = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(scores))
    emotion = SPEECH_EMOTIONS[idx]
    confidence = float(scores[idx])
    scores_dict = {emo: float(scores[i]) for i, emo in enumerate(SPEECH_EMOTIONS)}
    return emotion, confidence, scores_dict
