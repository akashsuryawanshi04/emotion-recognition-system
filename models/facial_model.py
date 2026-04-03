"""
models/facial_model.py
MiniXception CNN for Facial Emotion Recognition.
Trained on FER2013 dataset (48x48 grayscale face images).
"""

import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Add
)
from tensorflow.keras.optimizers import Adam

FACIAL_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 48
NUM_CLASSES = 7


def build_facial_model():
    """
    MiniXception architecture for facial emotion recognition.
    Uses depthwise separable convolutions + residual connections.
    Input : (48, 48, 1) grayscale face image
    Output: 7-class softmax probability vector
    """
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='face_input')

    # Initial conv block
    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(8, (3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual separable conv blocks
    for num_filters in [16, 32, 64, 128]:
        residual = Conv2D(num_filters, (1, 1), strides=(2, 2),
                          padding='same', use_bias=False)(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(num_filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = SeparableConv2D(num_filters, (3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = Add()([x, residual])

    # Output head
    x = Conv2D(NUM_CLASSES, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Activation('softmax', name='facial_output')(x)

    model = Model(inputs, outputs, name='MiniXception_Facial')
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def load_facial_model(path='saved_models/facial_emotion_model.h5'):
    return load_model(path)


def preprocess_face(face_img):
    """
    Preprocess a face image for model input.
    face_img: numpy array (any size, BGR or grayscale)
    Returns : numpy array (1, 48, 48, 1) normalized float32
    """
    import cv2
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=-1)
    face_img = np.expand_dims(face_img, axis=0)
    return face_img


def predict_facial_emotion(model, face_img):
    """
    Predict emotion from a face image.
    Returns: (emotion_label, confidence, all_scores_dict)
    """
    processed = preprocess_face(face_img)
    scores = model.predict(processed, verbose=0)[0]
    idx = int(np.argmax(scores))
    emotion = FACIAL_EMOTIONS[idx]
    confidence = float(scores[idx])
    scores_dict = {emo: float(scores[i]) for i, emo in enumerate(FACIAL_EMOTIONS)}
    return emotion, confidence, scores_dict
