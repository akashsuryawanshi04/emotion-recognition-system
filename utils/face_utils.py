"""
utils/face_utils.py
Face detection using OpenCV Haar Cascade.
Detects face region from a frame and returns cropped face.
"""

import cv2
import numpy as np
import os

# Load OpenCV's pre-trained face detector
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_faces(frame):
    """
    Detect all faces in a frame.
    frame : BGR numpy array (from cv2.VideoCapture)
    Returns: list of (x, y, w, h) bounding boxes
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return list(faces) if len(faces) > 0 else []


def get_largest_face(faces):
    """
    From multiple detected faces, return the largest one (main speaker).
    """
    if not faces:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def crop_face(frame, bbox, padding=10):
    """
    Crop face region from frame with optional padding.
    bbox : (x, y, w, h)
    Returns: cropped face numpy array or None
    """
    x, y, w, h = bbox
    h_frame, w_frame = frame.shape[:2]

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_frame, x + w + padding)
    y2 = min(h_frame, y + h + padding)

    face = frame[y1:y2, x1:x2]
    return face if face.size > 0 else None


def draw_emotion_box(frame, bbox, emotion, confidence, color=(147, 77, 255)):
    """
    Draw bounding box and emotion label on frame.
    color: BGR tuple (default purple)
    Returns: annotated frame
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"{emotion} {confidence * 100:.0f}%"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(y - 10, label_size[1] + 10)

    cv2.rectangle(frame,
                  (x, label_y - label_size[1] - 8),
                  (x + label_size[0] + 8, label_y + 4),
                  color, -1)
    cv2.putText(frame, label,
                (x + 4, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
