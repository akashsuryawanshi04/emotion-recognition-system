"""
app.py
Main Streamlit Application — Emotion Recognition from Speech & Facial Expressions

Features:
  - Mode 1: Video Recording Upload → analyze full video, show results + timeline
  - Mode 2: Live Webcam → real-time facial + speech emotion detection

Run:
    streamlit run app.py
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import threading
import time
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tensorflow.keras.models import load_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.facial_model import predict_facial_emotion, FACIAL_EMOTIONS
from models.speech_model import predict_speech_emotion, SPEECH_EMOTIONS
from models.fusion import fuse_predictions, fuse_video_results
from utils.face_utils import detect_faces, get_largest_face, crop_face, draw_emotion_box
from utils.audio_utils import extract_mfcc, extract_mfcc_from_video
from utils.video_utils import extract_frames, get_video_info

# ─────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

FACIAL_MODEL_PATH = 'saved_models/facial_emotion_model.h5'
SPEECH_MODEL_PATH = 'saved_models/speech_emotion_model.h5'

EMOTION_COLORS = {
    'Happy':    '#7F77DD',
    'Sad':      '#378ADD',
    'Angry':    '#D85A30',
    'Fear':     '#BA7517',
    'Disgust':  '#D4537E',
    'Surprise': '#1D9E75',
    'Neutral':  '#888780'
}

EMOTION_EMOJI = {
    'Happy': '😊', 'Sad': '😢', 'Angry': '😠',
    'Fear': '😨', 'Disgust': '🤢', 'Surprise': '😲', 'Neutral': '😐'
}

# ─────────────────────────────────────────────────────
#  LOAD MODELS (cached)
# ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    facial_model = None
    speech_model = None
    errors = []

    if os.path.exists(FACIAL_MODEL_PATH):
        try:
            facial_model = load_model(FACIAL_MODEL_PATH)
        except Exception as e:
            errors.append(f"Facial model error: {e}")
    else:
        errors.append(f"Facial model not found at '{FACIAL_MODEL_PATH}'. Run train/train_facial.py first.")

    if os.path.exists(SPEECH_MODEL_PATH):
        try:
            speech_model = load_model(SPEECH_MODEL_PATH)
        except Exception as e:
            errors.append(f"Speech model error: {e}")
    else:
        errors.append(f"Speech model not found at '{SPEECH_MODEL_PATH}'. Run train/train_speech.py first.")

    return facial_model, speech_model, errors


# ─────────────────────────────────────────────────────
#  HELPER: Emotion scores bar chart
# ─────────────────────────────────────────────────────
def render_emotion_bars(scores: dict, title="Emotion Scores"):
    emotions = list(scores.keys())
    values   = [scores[e] * 100 for e in emotions]
    colors   = [EMOTION_COLORS.get(e, '#aaaaaa') for e in emotions]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bars = ax.barh(emotions, values, color=colors, height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va='center', fontsize=8)
    plt.tight_layout()
    return fig


def render_emotion_timeline(timeline_data: list):
    """Plot emotion timeline across video frames."""
    if not timeline_data:
        return None

    timestamps = [t for t, _, _ in timeline_data]
    emotions   = [e for _, e, _ in timeline_data]

    emo_nums = {e: i for i, e in enumerate(FACIAL_EMOTIONS)}
    y_vals   = [emo_nums.get(e, 0) for e in emotions]
    clrs     = [EMOTION_COLORS.get(e, '#aaa') for e in emotions]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.scatter(timestamps, y_vals, c=clrs, s=18, alpha=0.85, zorder=3)
    ax.plot(timestamps, y_vals, color='#cccccc', linewidth=0.6, zorder=2)
    ax.set_yticks(range(len(FACIAL_EMOTIONS)))
    ax.set_yticklabels(FACIAL_EMOTIONS, fontsize=9)
    ax.set_xlabel("Time (seconds)", fontsize=9)
    ax.set_title("Emotion Timeline", fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────
def sidebar(facial_model, speech_model, errors):
    with st.sidebar:
        st.markdown("## 🎭 Emotion Recognition")
        st.markdown("*From Speech and Facial Expression*")
        st.divider()

        st.markdown("### Model Status")
        if facial_model is not None:
            st.success("✅ Facial model loaded")
        else:
            st.error("❌ Facial model missing")

        if speech_model is not None:
            st.success("✅ Speech model loaded")
        else:
            st.error("❌ Speech model missing")

        if errors:
            st.divider()
            st.markdown("### ⚠️ Setup Required")
            for err in errors:
                st.warning(err)
            st.info("Run training scripts first:\n"
                    "```\npython train/train_facial.py\n"
                    "python train/train_speech.py\n```")

        st.divider()
        st.markdown("### About")
        st.markdown("""
        **Models used:**
        - Facial: MiniXception CNN
        - Speech: CNN + LSTM
        - Fusion: Weighted Average

        **Datasets:**
        - FER2013 (facial)
        - RAVDESS (speech)

        **Emotions detected:**
        😊 Happy · 😢 Sad · 😠 Angry
        😨 Fear · 🤢 Disgust · 😲 Surprise · 😐 Neutral
        """)


# ─────────────────────────────────────────────────────
#  MODE 1: VIDEO UPLOAD
# ─────────────────────────────────────────────────────
def video_upload_mode(facial_model, speech_model):
    st.header("📹 Video Recording Analysis")
    st.markdown("Upload a video file and the system will analyse facial expressions and speech emotion throughout the video.")

    uploaded = st.file_uploader(
        "Upload video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )

    if uploaded is None:
        st.info("👆 Upload a video file to begin analysis.")
        return

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Show video info
    try:
        info = get_video_info(tmp_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Duration", f"{info['duration_sec']:.1f}s")
        col2.metric("FPS", f"{info['fps']:.0f}")
        col3.metric("Resolution", f"{info['width']}×{info['height']}")
        col4.metric("Total Frames", f"{info['total_frames']}")
    except:
        st.warning("Could not read video metadata.")

    st.divider()

    if st.button("🔍 Analyse Video", type="primary", use_container_width=True):
        if facial_model is None and speech_model is None:
            st.error("No models loaded. Please train models first.")
            return

        run_video_analysis(tmp_path, facial_model, speech_model)

    os.unlink(tmp_path)


def run_video_analysis(video_path, facial_model, speech_model):
    timeline_data = []
    face_results  = []
    speech_result = None

    progress = st.progress(0, text="Starting analysis...")
    status   = st.empty()

    # ── Step 1: Extract audio & run speech model ─────
    if speech_model is not None:
        status.text("🎙 Extracting audio and running speech model...")
        try:
            mfcc = extract_mfcc_from_video(video_path)
            s_emo, s_conf, s_scores = predict_speech_emotion(speech_model, mfcc)
            speech_result = (s_emo, s_conf, s_scores)
            status.text(f"🎙 Speech emotion: {s_emo} ({s_conf*100:.1f}%)")
        except Exception as e:
            st.warning(f"Speech analysis failed: {e}")

    progress.progress(25, text="Audio analysis done")

    # ── Step 2: Process video frames ─────────────────
    if facial_model is not None:
        status.text("🎭 Analysing facial expressions frame by frame...")
        try:
            total_frames = get_video_info(video_path)['total_frames']
            processed = 0

            for frame_idx, timestamp, frame in extract_frames(video_path, frame_skip=5):
                faces = detect_faces(frame)
                face = get_largest_face(faces)

                if face is not None:
                    x, y, w, h = face
                    face_crop = crop_face(frame, face)
                    if face_crop is not None:
                        f_emo, f_conf, f_scores = predict_facial_emotion(facial_model, face_crop)
                        face_results.append((f_emo, f_conf, f_scores))

                        if speech_result:
                            _, _, s_scores = speech_result
                            fused_emo, fused_conf, fused_scores = fuse_predictions(f_scores, s_scores)
                        else:
                            fused_emo, fused_conf, fused_scores = f_emo, f_conf, f_scores

                        timeline_data.append((timestamp, fused_emo, fused_scores))

                processed += 1
                pct = 25 + int((frame_idx / max(total_frames, 1)) * 70)
                progress.progress(min(pct, 95), text=f"Processing frames... {frame_idx}/{total_frames}")

        except Exception as e:
            st.error(f"Video analysis error: {e}")

    progress.progress(100, text="Analysis complete!")
    status.empty()

    # ── Step 3: Show results ──────────────────────────
    st.divider()
    st.subheader("📊 Results")

    if not face_results and speech_result is None:
        st.warning("No faces detected and no speech analysed.")
        return

    # Dominant emotion from fusion
    if face_results:
        all_fused_scores = {}
        for _, _, sc in timeline_data:
            for emo, val in sc.items():
                all_fused_scores.setdefault(emo, []).append(val)
        avg_fused = {e: float(np.mean(v)) for e, v in all_fused_scores.items()}
        dominant = max(avg_fused, key=avg_fused.get)
        dom_conf = avg_fused[dominant]
    elif speech_result:
        _, s_scores_dict = speech_result[1], speech_result[2]
        avg_fused = s_scores_dict
        dominant  = speech_result[0]
        dom_conf  = speech_result[1]
    else:
        return

    # Big result card
    col_main, col_face, col_speech = st.columns([1.5, 1, 1])

    with col_main:
        st.markdown(f"""
        <div style="background:#f8f8f8;border-radius:12px;padding:24px;text-align:center;
                    border:2px solid {EMOTION_COLORS.get(dominant,'#aaa')};">
            <div style="font-size:48px;">{EMOTION_EMOJI.get(dominant,'🎭')}</div>
            <div style="font-size:24px;font-weight:600;color:{EMOTION_COLORS.get(dominant,'#333')};
                        margin:8px 0;">{dominant}</div>
            <div style="font-size:14px;color:#666;">Dominant emotion · {dom_conf*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col_face:
        if face_results:
            all_face_scores = {}
            for _, _, sc in face_results:
                for emo, val in sc.items():
                    all_face_scores.setdefault(emo, []).append(val)
            avg_face = {e: float(np.mean(v)) for e, v in all_face_scores.items()}
            fig = render_emotion_bars(avg_face, "Facial Model")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col_speech:
        if speech_result:
            fig = render_emotion_bars(speech_result[2], "Speech Model")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # Stats row
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Faces detected", len(face_results))
    m2.metric("Frames analysed", len(timeline_data))
    m3.metric("Dominant emotion", dominant)
    m4.metric("Fusion confidence", f"{dom_conf*100:.1f}%")

    # Timeline
    if timeline_data:
        st.subheader("⏱ Emotion Timeline")
        fig_tl = render_emotion_timeline(timeline_data)
        if fig_tl:
            st.pyplot(fig_tl, use_container_width=True)
            plt.close()

    # Fusion scores table
    st.subheader("🔗 Fusion Scores (average across video)")
    fusion_cols = st.columns(len(FACIAL_EMOTIONS))
    for i, emo in enumerate(FACIAL_EMOTIONS):
        score = avg_fused.get(emo, 0.0)
        fusion_cols[i].metric(
            f"{EMOTION_EMOJI.get(emo,'')} {emo}",
            f"{score*100:.1f}%"
        )


# ─────────────────────────────────────────────────────
#  MODE 2: LIVE WEBCAM
# ─────────────────────────────────────────────────────
def live_webcam_mode(facial_model, speech_model):
    st.header("🔴 Live Webcam Detection")
    st.markdown("Real-time emotion detection using your webcam and microphone simultaneously.")

    if facial_model is None:
        st.error("Facial model not loaded. Cannot run live detection.")
        return

    col_feed, col_results = st.columns([1.3, 1])

    with col_feed:
        st.markdown("**Webcam Feed**")
        frame_window = st.image([], use_container_width=True)
        status_text  = st.empty()

    with col_results:
        st.markdown("**Live Predictions**")
        face_bar_chart  = st.empty()
        st.divider()
        st.markdown("**Fusion Result**")
        result_box      = st.empty()
        metric_cols     = st.columns(3)
        face_metric     = metric_cols[0].empty()
        speech_metric   = metric_cols[1].empty()
        fusion_metric   = metric_cols[2].empty()

    # Session state
    if 'live_running' not in st.session_state:
        st.session_state.live_running = False
    if 'speech_result_live' not in st.session_state:
        st.session_state.speech_result_live = None

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start = st.button("▶ Start Detection", type="primary", use_container_width=True)
    with btn_col2:
        stop  = st.button("⏹ Stop Detection", use_container_width=True)

    if start:
        st.session_state.live_running = True
    if stop:
        st.session_state.live_running = False

    if not st.session_state.live_running:
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder_img, "Press Start to begin",
                    (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        frame_window.image(placeholder_img, channels='BGR', use_container_width=True)
        return

    # ── Audio thread: updates speech prediction in background ──
    def audio_loop():
        try:
            from utils.audio_utils import record_audio_chunk, extract_mfcc
        except ImportError:
            return

        while st.session_state.get('live_running', False):
            y, sr = record_audio_chunk(duration=1.5)
            if y is not None and speech_model is not None:
                try:
                    mfcc = extract_mfcc(y=y, sr=sr)
                    s_emo, s_conf, s_scores = predict_speech_emotion(speech_model, mfcc)
                    st.session_state.speech_result_live = (s_emo, s_conf, s_scores)
                except Exception:
                    pass
            time.sleep(0.1)

    audio_thread = threading.Thread(target=audio_loop, daemon=True)
    audio_thread.start()

    # ── Main webcam loop ──────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam. Make sure camera is connected and allowed.")
        st.session_state.live_running = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0

    try:
        while st.session_state.live_running:
            ret, frame = cap.read()
            if not ret:
                status_text.warning("Cannot read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)  # mirror
            display_frame = frame.copy()
            frame_count += 1

            # Predict every 3rd frame for performance
            if frame_count % 3 == 0:
                faces = detect_faces(frame)
                face  = get_largest_face(faces)

                if face is not None:
                    x, y, w, h = face
                    face_crop = crop_face(frame, face)

                    if face_crop is not None:
                        f_emo, f_conf, f_scores = predict_facial_emotion(facial_model, face_crop)

                        s_result = st.session_state.speech_result_live
                        if s_result and speech_model is not None:
                            s_emo, s_conf, s_scores = s_result
                            fused_emo, fused_conf, fused_scores = fuse_predictions(f_scores, s_scores)
                        else:
                            fused_emo, fused_conf, fused_scores = f_emo, f_conf, f_scores
                            s_conf = 0.0
                            s_emo  = 'N/A'

                        # Annotate frame
                        color_bgr = tuple(int(EMOTION_COLORS.get(fused_emo, '#888')[i:i+2], 16)
                                          for i in (5, 3, 1))
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color_bgr, 2)
                        label = f"{fused_emo} {fused_conf*100:.0f}%"
                        cv2.rectangle(display_frame, (x, y-32), (x+w, y), color_bgr, -1)
                        cv2.putText(display_frame, label, (x+4, y-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                        # Update UI
                        fig = render_emotion_bars(fused_scores, "Fusion Scores")
                        face_bar_chart.pyplot(fig, use_container_width=True)
                        plt.close()

                        result_box.markdown(f"""
                        <div style="padding:14px;background:#f5f5f5;border-radius:10px;
                                    border-left:4px solid {EMOTION_COLORS.get(fused_emo,'#aaa')};">
                            <span style="font-size:28px;">{EMOTION_EMOJI.get(fused_emo,'🎭')}</span>
                            <span style="font-size:20px;font-weight:600;margin-left:10px;
                                         color:{EMOTION_COLORS.get(fused_emo,'#333')}">{fused_emo}</span>
                            <div style="font-size:12px;color:#888;margin-top:4px;">
                                Fusion confidence: {fused_conf*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        face_metric.metric("Face", f"{f_conf*100:.0f}%", delta=f_emo)
                        speech_metric.metric("Speech", f"{s_conf*100:.0f}%" if isinstance(s_conf, float) else "—", delta=s_emo if s_emo != 'N/A' else None)
                        fusion_metric.metric("Fusion", f"{fused_conf*100:.0f}%", delta=fused_emo)

                else:
                    cv2.putText(display_frame, "No face detected",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            # Show FPS & REC tag
            cv2.circle(display_frame, (620, 20), 8, (0, 0, 220), -1)
            cv2.putText(display_frame, "LIVE", (582, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            frame_window.image(display_frame, channels='BGR', use_container_width=True)

    finally:
        cap.release()
        st.session_state.live_running = False


# ─────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────
def main():
    # Load models
    with st.spinner("Loading emotion recognition models..."):
        facial_model, speech_model, errors = load_models()

    # Sidebar
    sidebar(facial_model, speech_model, errors)

    # Title
    st.title("🎭 Emotion Recognition System")
    st.markdown("*Speech & Facial Expression Analysis*")
    st.divider()

    # Mode selection tabs
    tab1, tab2 = st.tabs(["📹 Video Recording", "🔴 Live Webcam"])

    with tab1:
        video_upload_mode(facial_model, speech_model)

    with tab2:
        live_webcam_mode(facial_model, speech_model)


if __name__ == '__main__':
    main()
