"""
utils/audio_utils.py
Audio feature extraction for Speech Emotion Recognition.
Extracts MFCC features from audio files or raw waveforms.
"""

import numpy as np
import librosa
import soundfile as sf
import os
import tempfile

N_MFCC = 40
MAX_FRAMES = 200
SAMPLE_RATE = 22050


def extract_mfcc(audio_path=None, y=None, sr=None):
    """
    Extract MFCC features from audio file or raw waveform.

    Args:
        audio_path: path to .wav/.mp3 file (optional)
        y         : raw waveform numpy array (optional)
        sr        : sample rate (required if y is provided)

    Returns:
        mfcc: numpy array of shape (N_MFCC, MAX_FRAMES)
    """
    if audio_path is not None:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    elif y is None:
        raise ValueError("Provide either audio_path or raw waveform y")

    # Extract MFCC (40 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Pad or trim to MAX_FRAMES
    if mfcc.shape[1] < MAX_FRAMES:
        pad_width = MAX_FRAMES - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    return mfcc.astype(np.float32)


def extract_audio_from_video(video_path, output_audio_path=None):
    """
    Extract audio track from a video file using ffmpeg.
    Returns path to extracted .wav file.
    """
    import subprocess

    if output_audio_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_audio_path = tmp.name
        tmp.close()

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', str(SAMPLE_RATE),
        '-ac', '1',
        output_audio_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to extract audio from {video_path}")

    return output_audio_path


def extract_mfcc_from_video(video_path):
    """
    Extract MFCC features directly from a video file.
    Returns: mfcc numpy array (N_MFCC, MAX_FRAMES)
    """
    audio_path = extract_audio_from_video(video_path)
    try:
        mfcc = extract_mfcc(audio_path=audio_path)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
    return mfcc


def record_audio_chunk(duration=1.5, sr=SAMPLE_RATE):
    """
    Record a short audio chunk from microphone.
    Used in live webcam mode for real-time speech analysis.
    Returns: raw waveform numpy array
    """
    try:
        import sounddevice as sd
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        return audio.flatten(), sr
    except Exception as e:
        print(f"Microphone recording failed: {e}")
        return None, None
