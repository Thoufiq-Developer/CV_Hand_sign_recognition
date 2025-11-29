import streamlit as st
import threading
import time
import cv2
import numpy as np
from PIL import Image
import io

try:
    import mediapipe as mp
except Exception:
    mp = None

st.set_page_config(page_title="Live Hand Sign Recognition", layout="wide")
st.title("Live Hand Sign Recognition")

if mp is None:
    st.error("mediapipe is not installed. This app requires mediapipe for live webcam analysis. Install mediapipe and restart the app.")
    st.stop()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def calc_dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

def detect_signs(landmarks):
    try:
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

        thumb_extended = thumb_tip.x < thumb_ip.x if True else thumb_tip.y < thumb_ip.y
        fingers_extended = (
            index_tip.y < index_mcp.y,
            middle_tip.y < middle_mcp.y,
            ring_tip.y < ring_mcp.y,
            pinky_tip.y < pinky_mcp.y
        )

        if thumb_tip.y > middle_tip.y and calc_dist(middle_tip, ring_tip) > calc_dist(index_tip, middle_tip):
            return "Vulcan / Fire signal"
        if all(fingers_extended) and thumb_extended:
            return "Open Hand / Help"
        if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y:
            return "Medical Alert"
        if thumb_extended and (index_tip.y < index_mcp.y):
            if middle_tip.y > thumb_tip.y and ring_tip.y > thumb_tip.y and pinky_tip.y > thumb_tip.y:
                return "Brake Fail"
        return None
    except Exception:
        return None

if "running" not in st.session_state:
    st.session_state.running = False
if "last_label" not in st.session_state:
    st.session_state.last_label = "None"
if "counters" not in st.session_state:
    st.session_state.counters = {"Vulcan / Fire signal":0,"Open Hand / Help":0,"Medical Alert":0,"Brake Fail":0,"Unknown":0}

col1, col2 = st.columns([3,1])
with col1:
    st.header("Live Video")
    mode = st.radio("Mode", ["Webcam (Live)", "Upload Image"], index=0)
    video_placeholder = st.empty()
    start = st.button("Start") if mode=="Webcam (Live)" else None
    stop = st.button("Stop") if mode=="Webcam (Live)" else None

with col2:
    st.header("Analysis")
    st.write("Last detected:")
    last_box = st.empty()
    st.write("Counters:")
    count_box = st.empty()
    if st.button("Reset Counters"):
        st.session_state.counters = {k:0 for k in st.session_state.counters}
        st.session_state.last_label = "None"

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

def process_frame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    label = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = detect_signs(hand_landmarks.landmark)
            if label is None:
                label = "Unknown"
            cv2.putText(frame_bgr, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,0), 2, cv2.LINE_AA)
    return frame_bgr, label

def webcam_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to open webcam. Make sure camera is free and accessible.")
        st.session_state.running = False
        return
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        annotated, label = process_frame(frame)
        if label:
            st.session_state.last_label = label
            st.session_state.counters[label] = st.session_state.counters.get(label,0)+1
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_rgb, channels="RGB")
        last_box.markdown(f"**{st.session_state.last_label}**")
        count_box.table(st.session_state.counters)
        time.sleep(0.03)
    cap.release()

if mode=="Webcam (Live)":
    if start and not st.session_state.running:
        st.session_state.running = True
        t = threading.Thread(target=webcam_loop, daemon=True)
        t.start()
    if stop and st.session_state.running:
        st.session_state.running = False

if mode=="Upload Image":
    uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        frame = np.array(image)[:,:,::-1].copy()
        annotated, label = process_frame(frame)
        if label:
            st.session_state.last_label = label
            st.session_state.counters[label] = st.session_state.counters.get(label,0)+1
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_rgb, channels="RGB")
        last_box.markdown(f"**{st.session_state.last_label}**")
        count_box.table(st.session_state.counters)

st.markdown("---")

