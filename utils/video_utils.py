"""
utils/video_utils.py
Video frame extraction and processing pipeline.
"""

import cv2
import numpy as np
import os


def extract_frames(video_path, frame_skip=5):
    """
    Extract frames from a video file.
    frame_skip: process every Nth frame (5 = 6 fps from 30 fps video)

    Yields: (frame_index, frame_timestamp_sec, frame_BGR_array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            timestamp = frame_idx / fps
            yield frame_idx, timestamp, frame
        frame_idx += 1

    cap.release()


def get_video_info(video_path):
    """Return basic info about a video file."""
    cap = cv2.VideoCapture(video_path)
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    }
    cap.release()
    return info


def annotate_frame(frame, emotion, confidence, face_bbox=None):
    """Draw emotion label and bounding box on a frame."""
    annotated = frame.copy()

    if face_bbox is not None:
        x, y, w, h = face_bbox
        color = emotion_color(emotion)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label = f"{emotion} {confidence * 100:.0f}%"
        cv2.putText(annotated, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return annotated


def emotion_color(emotion):
    """Return BGR color for each emotion."""
    colors = {
        'Happy': (77, 127, 255),
        'Sad': (221, 90, 56),
        'Angry': (48, 30, 216),
        'Fear': (23, 117, 186),
        'Disgust': (126, 83, 212),
        'Surprise': (75, 158, 29),
        'Neutral': (128, 135, 136)
    }
    return colors.get(emotion, (180, 180, 180))
