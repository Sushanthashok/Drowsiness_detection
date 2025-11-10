# app.py
import os
import streamlit as st
import numpy as np
from tempfile import NamedTemporaryFile

# ========== Environment Fixes ==========
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"      # Fix hanging on some systems
os.environ["GLOG_minloglevel"] = "2"           # Silence MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # Silence TensorFlow logs

# ========== Page Setup ==========
st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("😴 Drowsiness Detection Model")
st.caption("Detects sleeping (red) and awake (green) persons in images/videos, with optional age prediction.")
st.write("---")

# ========== Lazy-load Libraries ==========
@st.cache_resource
def load_cv2():
    import cv2
    return cv2

@st.cache_resource
def load_face_mesh():
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    try:
        return mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
    except Exception as e:
        st.warning(f"⚠️ Fallback FaceMesh loaded (no refine): {e}")
        return mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

@st.cache_resource
def try_load_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception as e:
        st.warning(f"⚠️ Age prediction disabled: {e}")
        return None

cv2 = load_cv2()
face_mesh = load_face_mesh()
DeepFace = try_load_deepface()

from utils.vision import process_frame

# ========== Controls (main area instead of sidebar) ==========
col1, col2, col3 = st.columns(3)
with col1:
    mode = st.radio("Select Mode", ["Image", "Video"])
with col2:
    ear_thresh = st.slider("EAR Threshold (lower = stricter)", 0.15, 0.40, 0.28, 0.01)
with col3:
    enable_age = st.checkbox("Enable Age Prediction", value=True)

st.write("---")

# ========== IMAGE MODE ==========
if mode == "Image":
    file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])
    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.subheader("Preview")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.info("Running prediction...")
        annotated, stats = process_frame(
            img_bgr.copy(),
            face_mesh,
            ear_thresh=ear_thresh,
            age_box_scale=1.2,
            DeepFace=DeepFace if enable_age else None
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Output")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        with c2:
            st.subheader("Summary")
            st.metric("People Detected", stats["total"])
            st.metric("Sleeping", stats["sleeping"])
            if stats["sleeping"] > 0:
                ages_text = ", ".join(map(str, stats["sleepers_ages"])) if stats["sleepers_ages"] else "?"
                st.write(f"**Ages (sleepers):** {ages_text}")

        # Pop-up message
        msg = f"Sleeping: {stats['sleeping']} / {stats['total']}"
        if stats["sleeping"] > 0 and stats["sleepers_ages"]:
            msg += f" | Ages: {', '.join(map(str, stats['sleepers_ages']))}"
        st.toast(msg, icon="⚠️" if stats["sleeping"] > 0 else "✅")

# ========== VIDEO MODE ==========
else:
    file = st.file_uploader("🎥 Upload a Video", type=["mp4", "mov", "avi", "mkv"])
    if file:
        with NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            video_path = tmp.name

        st.subheader("Preview & Output")
        frame_holder = st.empty()
        info_holder = st.empty()

        cap = cv2.VideoCapture(video_path)
        total_sleeping = 0
        total_people = 0
        aggregated_ages = []
        frame_interval = 3
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_interval != 0:
                idx += 1
                continue

            annotated, stats = process_frame(
                frame,
                face_mesh,
                ear_thresh=ear_thresh,
                age_box_scale=1.2,
                DeepFace=DeepFace if enable_age else None
            )

            total_people = max(total_people, stats["total"])
            total_sleeping = max(total_sleeping, stats["sleeping"])
            aggregated_ages.extend(stats["sleepers_ages"])

            frame_holder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            info_holder.info(f"Detected: {stats['total']} | Sleeping: {stats['sleeping']}")
            idx += 1

        cap.release()
        ages_text = ", ".join(map(str, aggregated_ages)) if aggregated_ages else "?"
        st.toast(f"🎯 Summary → Max Sleeping: {total_sleeping}/{total_people} | Ages: {ages_text}")
