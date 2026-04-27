import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
import subprocess
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- CONFIGURATION ---
MODEL_DIR = 'models'
MODELS = {
    'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
    'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task',
    'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
}

# Drawing Connections
POSE_CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(15,17),(15,19),(15,21),(16,18),(16,20),(16,22)]
HAND_CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, url in MODELS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            with st.spinner(f"Downloading {fname}..."):
                urllib.request.urlretrieve(url, path)

def get_available_cameras():
    """Detect available video devices on Linux."""
    try:
        output = subprocess.check_output("ls /dev/video*", shell=True).decode().split()
        return output
    except:
        return []

def draw_landmarks(image, pose_res, face_res, hand_res):
    h, w = image.shape[:2]
    ann = image.copy()

    if pose_res and pose_res.pose_landmarks:
        for landmarks in pose_res.pose_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in POSE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(ann, pts[a], pts[b], (0, 220, 0), 2, cv2.LINE_AA)
            for pt in pts: cv2.circle(ann, pt, 4, (0, 255, 0), -1, cv2.LINE_AA)

    if face_res and face_res.face_landmarks:
        for landmarks in face_res.face_landmarks:
            for lm in landmarks:
                cv2.circle(ann, (int(lm.x * w), int(lm.y * h)), 1, (0, 165, 255), -1)

    if hand_res and hand_res.hand_landmarks:
        for landmarks in hand_res.hand_landmarks:
            hpts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
            for a, b in HAND_CONNECTIONS:
                if a < len(hpts) and b < len(hpts):
                    cv2.line(ann, hpts[a], hpts[b], (60, 60, 255), 2, cv2.LINE_AA)
            for pt in hpts: cv2.circle(ann, pt, 5, (80, 80, 255), -1, cv2.LINE_AA)
    
    return ann

def main():
    st.set_page_config(page_title="Holistic Vision | SLT Baseline", page_icon="🤟", layout="wide")
    download_models()

    st.title("Holistic Vision 🤟")
    
    # Diagnostics in Sidebar
    st.sidebar.title("🛠 Diagnostics")
    cameras = get_available_cameras()
    if cameras:
        st.sidebar.success(f"Found {len(cameras)} video devices: {', '.join(cameras)}")
    else:
        st.sidebar.error("❌ No video devices found in /dev/video*. Are you running this on a remote server?")

    cam_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0)
    run_demo = st.sidebar.checkbox("🚀 Start Live Stream", value=False)

    if run_demo:
        cap = cv2.VideoCapture(cam_index)
        
        # Initialize Detectors
        base_options_pose = mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, 'pose_landmarker.task'))
        options_pose = mp_vision.PoseLandmarkerOptions(base_options=base_options_pose, running_mode=mp_vision.RunningMode.IMAGE)
        
        base_options_face = mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, 'face_landmarker.task'))
        options_face = mp_vision.FaceLandmarkerOptions(base_options=base_options_face, running_mode=mp_vision.RunningMode.IMAGE)
        
        base_options_hand = mp_python.BaseOptions(model_asset_path=os.path.join(MODEL_DIR, 'hand_landmarker.task'))
        options_hand = mp_vision.HandLandmarkerOptions(base_options=base_options_hand, running_mode=mp_vision.RunningMode.IMAGE, num_hands=2)

        with mp_vision.PoseLandmarker.create_from_options(options_pose) as pose_det, \
             mp_vision.FaceLandmarker.create_from_options(options_face) as face_det, \
             mp_vision.HandLandmarker.create_from_options(options_hand) as hand_det:

            FRAME_WINDOW = st.image([], width=700) # Fixed width to avoid warning if stretch fails
            
            if not cap.isOpened():
                st.error(f"Cannot open camera at index {cam_index}")
            else:
                while cap.isOpened() and run_demo:
                    ret, frame = cap.read()
                    if not ret: break

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_frame))

                    pose_res = pose_det.detect(mp_img)
                    face_res = face_det.detect(mp_img)
                    hand_res = hand_det.detect(mp_img)

                    ann_frame = draw_landmarks(rgb_frame, pose_res, face_res, hand_res)
                    FRAME_WINDOW.image(ann_frame)

                cap.release()

if __name__ == "__main__":
    main()
