import streamlit as st
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Hand Sign Recognition", layout="wide")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5

def is_violence_at_home_hand(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    thumb_extended = thumb_tip.y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    fingers_extended = (
        index_tip.y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
        middle_tip.y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
        ring_tip.y < landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y and
        pinky_tip.y < landmarks[mp_hands.HandLandmark.PINKY_MCP].y
    )
    return thumb_extended and fingers_extended

def is_hand_open(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    distance_thumb_index = calculate_distance(thumb_tip, index_tip)
    distance_thumb_middle = calculate_distance(thumb_tip, middle_tip)
    return distance_thumb_index > 0.10 and distance_thumb_middle > 0.10

def fire_alert(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    hand_size = calculate_distance(landmarks[0], middle_tip)
    distance_middle_ring = calculate_distance(middle_tip, ring_tip)
    distance_index_middle = calculate_distance(index_tip, middle_tip)
    vulcan_salute = (thumb_tip.y > middle_tip.y and distance_middle_ring > distance_index_middle)
    return vulcan_salute

def medical_alert(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    return (index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and
            ring_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y)

def brake_fail(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    thumb_extended = thumb_tip.y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_extended = index_tip.y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    mrp_curled = (middle_tip.y > thumb_tip.y and middle_tip.y > index_tip.y and
                  pinky_tip.y > thumb_tip.y and pinky_tip.y > index_tip.y and
                  ring_tip.y > thumb_tip.y and ring_tip.y > index_tip.y)
    return thumb_extended and index_extended and mrp_curled

def detect_signs_from_landmarks(landmarks):
    try:
        if fire_alert(landmarks):
            return "Fire Signal"
        if is_violence_at_home_hand(landmarks):
            return "Help / Violence Signal"
        if is_hand_open(landmarks):
            return "Open Hand / Help"
        if medical_alert(landmarks):
            return "Medical Alert"
        if brake_fail(landmarks):
            return "Brake Fail"
    except Exception:
        return None
    return None

# UI layout
st.title("Hand Sign Recognition")
st.markdown("Live webcam and single-image detection. Analysis panel shows last detected label and counters.")

col1, col2 = st.columns([2, 1])
with col1:
    mode = st.radio("Mode", ["Webcam (live)", "Upload image"], index=0)
    img_placeholder = st.empty()
    start_button = st.button("Start Webcam") if mode == "Webcam (live)" else None
    stop_button = st.button("Stop Webcam") if mode == "Webcam (live)" else None

with col2:
    st.subheader("Analysis")
    last_label = st.empty()
    counts_area = st.empty()
    reset_counts = st.button("Reset counters")

# session state for background thread and counters
if "running" not in st.session_state:
    st.session_state.running = False
if "counters" not in st.session_state:
    st.session_state.counters = {"Fire Signal": 0, "Help / Violence Signal": 0, "Open Hand / Help": 0, "Medical Alert": 0, "Brake Fail": 0, "Unknown": 0}
if "last" not in st.session_state:
    st.session_state.last = "None"

if reset_counts:
    st.session_state.counters = {k: 0 for k in st.session_state.counters}
    st.session_state.last = "None"

hands_proc = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

def process_frame(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands_proc.process(img_rgb)
    label = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            label = detect_signs_from_landmarks(landmarks)
            mp_draw.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if label:
                cv2.putText(frame_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 200, 10), 2)
    return frame_bgr, label

# Webcam thread
def webcam_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state.running = False
        return
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        annotated, label = process_frame(frame)
        if label:
            st.session_state.last = label
            st.session_state.counters[label] = st.session_state.counters.get(label, 0) + 1
        # convert BGR->RGB for display
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img_placeholder.image(annotated_rgb, channels="RGB")
        last_label.markdown(f"**Last detected:** {st.session_state.last}")
        counts_area.table(st.session_state.counters)
        time.sleep(0.03)
    cap.release()

# Start / stop control
if mode == "Webcam (live)":
    if start_button and not st.session_state.running:
        st.session_state.running = True
        t = threading.Thread(target=webcam_thread, daemon=True)
        t.start()
    if stop_button and st.session_state.running:
        st.session_state.running = False

# Upload image mode
if mode == "Upload image":
    uploaded_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Cannot read image")
        else:
            annotated, label = process_frame(frame.copy())
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img_placeholder.image(annotated_rgb, channels="RGB")
            if label:
                st.session_state.last = label
                st.session_state.counters[label] = st.session_state.counters.get(label, 0) + 1
            last_label.markdown(f"**Last detected:** {st.session_state.last}")
            counts_area.table(st.session_state.counters)

st.markdown("---")
st.caption("Notes: Camera access works when running locally. On hosted platforms the webcam may not be accessible. Use the upload mode for testing images.")
