# 🎭 Emotion Recognition from Speech & Facial Expressions
**MCA Mega Project** | Python · TensorFlow · OpenCV · Streamlit

---

## 📁 Project Structure

```
emotion_recognition/
├── app.py                          ← Main Streamlit app (run this)
├── requirements.txt                ← All Python dependencies
│
├── train/
│   ├── train_facial.py             ← Train facial emotion model
│   └── train_speech.py             ← Train speech emotion model
│
├── models/
│   ├── facial_model.py             ← MiniXception CNN architecture
│   ├── speech_model.py             ← CNN + LSTM architecture
│   └── fusion.py                   ← Weighted fusion logic
│
├── utils/
│   ├── face_utils.py               ← Face detection (OpenCV)
│   ├── audio_utils.py              ← MFCC extraction (Librosa)
│   └── video_utils.py              ← Video frame extraction
│
├── data/                           ← Put datasets here
│   ├── fer2013/                    ← FER2013 dataset
│   │   ├── train/
│   │   │   ├── angry/
│   │   │   ├── disgust/
│   │   │   ├── fear/
│   │   │   ├── happy/
│   │   │   ├── neutral/
│   │   │   ├── sad/
│   │   │   └── surprise/
│   │   └── test/
│   │       └── (same structure as train/)
│   └── ravdess/                    ← RAVDESS dataset
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ... Actor_24/
│
└── saved_models/                   ← Trained models saved here (auto-created)
    ├── facial_emotion_model.h5
    ├── speech_emotion_model.h5
    ├── facial_confusion_matrix.png
    ├── facial_training_history.png
    ├── speech_confusion_matrix.png
    └── speech_training_history.png
```

---

## 🚀 Setup & Run — Step by Step

### Step 1 — Clone / Download this project
Place all files in a folder called `emotion_recognition/`.

---

### Step 2 — Install Python dependencies
```bash
pip install -r requirements.txt
```

> **Note:** Also install ffmpeg on your system:
> - **Windows:** Download from https://ffmpeg.org/download.html and add to PATH
> - **Ubuntu/Mac:** `sudo apt install ffmpeg` or `brew install ffmpeg`

---

### Step 3 — Download Datasets

#### 🗂 Dataset 1: FER2013 (for facial model)
1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Create a free Kaggle account if you don't have one
3. Download and extract the zip file
4. Place it so the structure looks like:
```
emotion_recognition/
└── data/
    └── fer2013/
        ├── train/
        │   ├── angry/       ← images of angry faces
        │   ├── happy/       ← images of happy faces
        │   └── ... (7 folders total)
        └── test/
            └── ... (same 7 folders)
```

#### 🗂 Dataset 2: RAVDESS (for speech model)
1. Go to: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
2. Download and extract the zip file
3. Place it so the structure looks like:
```
emotion_recognition/
└── data/
    └── ravdess/
        ├── Actor_01/
        │   ├── 03-01-01-01-01-01-01.wav
        │   └── ... (more .wav files)
        ├── Actor_02/
        └── ... Actor_24/
```

---

### Step 4 — Train the Facial Emotion Model
```bash
python train/train_facial.py
```
- Expected time: 30–90 minutes depending on your GPU/CPU
- Expected accuracy: **60–68%** (FER2013 is a hard dataset)
- Saves model to: `saved_models/facial_emotion_model.h5`
- Also saves: confusion matrix + training history graphs

---

### Step 5 — Train the Speech Emotion Model
```bash
python train/train_speech.py
```
- Expected time: 15–40 minutes
- Expected accuracy: **70–80%** on RAVDESS
- Saves model to: `saved_models/speech_emotion_model.h5`

---

### Step 6 — Run the Application
```bash
streamlit run app.py
```
- Opens at: http://localhost:8501
- Two modes available:
  - **📹 Video Recording** — Upload any .mp4 / .avi / .mov file
  - **🔴 Live Webcam** — Real-time detection from webcam + microphone

---

## 🧠 ML Models Used

| Model | Purpose | Dataset |
|---|---|---|
| MiniXception CNN | Facial emotion recognition | FER2013 |
| CNN + LSTM | Speech emotion recognition | RAVDESS |
| Weighted Average Fusion | Combine both model outputs | — |
| OpenCV Haar Cascade | Face detection in frames | Pre-trained |

## 🎭 Emotions Detected
`Happy` · `Sad` · `Angry` · `Fear` · `Disgust` · `Surprise` · `Neutral`

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ffmpeg not found` | Install ffmpeg and add to system PATH |
| `Cannot access webcam` | Allow camera permissions in browser |
| `Model not found` | Run training scripts in Step 4 and 5 |
| `CUDA out of memory` | Reduce BATCH_SIZE in training scripts |
| `No faces detected` | Ensure good lighting, face the camera directly |
| `Microphone not working` | Install `sounddevice`: `pip install sounddevice` |
