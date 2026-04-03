"""
models/fusion.py
Late fusion of facial and speech emotion predictions.
Combines probability scores from both models using weighted average.
"""

import numpy as np

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Weight for each modality — face is slightly more reliable
FACE_WEIGHT = 0.55
SPEECH_WEIGHT = 0.45


def fuse_predictions(face_scores: dict, speech_scores: dict,
                     face_weight=FACE_WEIGHT, speech_weight=SPEECH_WEIGHT):
    """
    Combine facial and speech emotion probability dicts using weighted average.

    Args:
        face_scores   : dict {emotion: probability} from facial model
        speech_scores : dict {emotion: probability} from speech model
        face_weight   : weight for facial model (default 0.55)
        speech_weight : weight for speech model (default 0.45)

    Returns:
        final_emotion : string — predicted emotion label
        confidence    : float — confidence of final prediction
        fused_scores  : dict  — combined probabilities for all emotions
    """
    fused = {}
    for emo in EMOTIONS:
        f_score = face_scores.get(emo, 0.0)
        s_score = speech_scores.get(emo, 0.0)
        fused[emo] = (face_weight * f_score) + (speech_weight * s_score)

    # Normalize so scores sum to 1
    total = sum(fused.values())
    if total > 0:
        fused = {emo: v / total for emo, v in fused.items()}

    final_emotion = max(fused, key=fused.get)
    confidence = fused[final_emotion]

    return final_emotion, confidence, fused


def fuse_video_results(frame_emotions: list):
    """
    Aggregate emotion predictions over all frames of a video.
    Returns the dominant emotion across the whole video.

    Args:
        frame_emotions: list of (emotion, confidence, fused_scores) tuples
    Returns:
        dominant_emotion: most common emotion in video
        avg_scores: average probability per emotion across all frames
    """
    if not frame_emotions:
        return 'Neutral', 0.0, {e: 0.0 for e in EMOTIONS}

    all_scores = {e: [] for e in EMOTIONS}
    for _, _, scores in frame_emotions:
        for emo in EMOTIONS:
            all_scores[emo].append(scores.get(emo, 0.0))

    avg_scores = {emo: float(np.mean(vals)) for emo, vals in all_scores.items()}
    dominant_emotion = max(avg_scores, key=avg_scores.get)
    confidence = avg_scores[dominant_emotion]

    return dominant_emotion, confidence, avg_scores
