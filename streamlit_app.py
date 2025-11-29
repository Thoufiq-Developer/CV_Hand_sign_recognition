import io
import time
import math
import json
from collections import Counter
from typing import Tuple
from PIL import Image
import numpy as np
import cv2
import streamlit as st

st.set_page_config(page_title="Hand Sign Detector (OpenCV)", layout="wide")

def read_image(uploaded_file) -> np.ndarray:
    data = uploaded_file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(image)

def preprocess_hand(img: np.ndarray) -> np.ndarray:
    img_blur = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 20, 70])
    upper = np.array([20, 255, 255])
    mask1 = cv2.inRange(hsv, lower, upper)
    lower2 = np.array([170,20,70])
    upper2 = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def largest_contour(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    return c, area

def count_fingers_from_contour(contour: np.ndarray, image: np.ndarray) -> Tuple[int, float, np.ndarray]:
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0, 0.0, image
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0, 0.0, image
    h, w = image.shape[:2]
    finger_count = 0
    annotated = image.copy()
    depths = []
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = math.dist(start, end)
        b = math.dist(start, far)
        c = math.dist(end, far)
        angle = math.acos(max(0.0, min(1.0, (b*b + c*c - a*a) / (2*b*c + 1e-8)))) * 180 / math.pi
        if angle < 90 and depth > 1000:
            finger_count += 1
            depths.append(depth)
            cv2.circle(annotated, far, 5, (0,255,0), -1)
            cv2.line(annotated, start, far, (255,0,0), 2)
            cv2.line(annotated, end, far, (255,0,0), 2)
    # fingers = defects + 1 if any defects else 0, but clamp
    fingers = min(5, finger_count + 1 if finger_count>0 else 0)
    conf = min(1.0, (sum(depths)/ (len(depths)+1)) / 5000.0) if depths else 0.0
    return fingers, conf, annotated

def detect_gesture(fingers: int, conf: float) -> str:
    if conf < 0.05:
        return "No hand detected / uncertain"
    if fingers == 0:
        return "Fist / No fingers"
    if fingers == 1:
        return "One finger"
    if fingers == 2:
        return "Two fingers"
    if fingers == 3:
        return "Three fingers"
    if fingers == 4:
        return "Four fingers"
    if fingers >= 5:
        return "Open hand (5+ fingers)"

# UI
st.markdown("# Hand Sign Detector")
st.markdown("Upload a clear hand image (good lighting, plain background) and the app will detect number of fingers and a basic gesture label.")
col_left, col_right = st.columns([2,1])

with col_left:
    uploaded = st.file_uploader("Upload image (jpg/png)", type=["jpg","jpeg","png"])
    st.markdown("Or drag & drop an image into this box.")
    if uploaded:
        img_np = read_image(uploaded)
        display_img = img_np.copy()
        st.image(display_img, caption="Input image", use_column_width=True)
        mask = preprocess_hand(img_np)
        contour, area = largest_contour(mask)
        if contour is None or area < 2000:
            st.warning("No prominent hand contour found. Try a clearer image or crop background.")
            detected_label = "No hand"
            fingers = 0
            conf = 0.0
            annotated = display_img
        else:
            fingers, conf, annotated = count_fingers_from_contour(contour, display_img)
            detected_label = detect_gesture(fingers, conf)
            # draw contour and bbox
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x,y), (x+w, y+h), (0,128,255), 2)
            cv2.drawContours(annotated, [contour], -1, (0,255,0), 2)
            st.image(annotated, caption="Annotated result", use_column_width=True)
        st.success(f"Detected: {detected_label}")
        st.markdown(f"- Fingers (estimated): **{fingers}**")
        st.markdown(f"- Confidence (approx): **{conf:.2f}**")
        # provide downloadable analysis JSON
        analysis = {
            "label": detected_label,
            "fingers_estimated": int(fingers),
            "confidence": float(conf),
            "contour_area": float(area)
        }
        btn = st.download_button("Download Analysis (JSON)", data=json.dumps(analysis, indent=2), file_name="hand_analysis.json", mime="application/json")

with col_right:
    st.subheader("Analysis Panel")
    st.write("This panel summarizes recent uploads.")
    if "history" not in st.session_state:
        st.session_state.history = []
    if uploaded:
        st.session_state.history.append({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "label": detected_label,
            "fingers": int(fingers),
            "confidence": float(conf),
            "area": float(area)
        })
    history = list(reversed(st.session_state.history))[:10]
    if history:
        st.table(history)
    else:
        st.write("No analyses yet. Upload an image to start.")

st.markdown("---")
