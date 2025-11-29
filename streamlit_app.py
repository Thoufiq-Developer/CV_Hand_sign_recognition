import io
import time
import math
import json
from typing import Tuple
from collections import deque
from PIL import Image
import numpy as np
import cv2
import streamlit as st

st.set_page_config(page_title="Hand Sign Detector (OpenCV)", layout="wide")

def read_imagefile(uploaded) -> np.ndarray:
    data = uploaded.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)

def preprocess_hand(img: np.ndarray) -> np.ndarray:
    img_small = cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1])))
    blur = cv2.GaussianBlur(img_small, (7, 7), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 20, 70])
    upper1 = np.array([20, 255, 255])
    lower2 = np.array([170,20,70])
    upper2 = np.array([180,255,255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask, img_small

def largest_contour(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0
    c = max(cnts, key=cv2.contourArea)
    return c, cv2.contourArea(c)

def count_fingers_from_contour(contour: np.ndarray, img: np.ndarray) -> Tuple[int, float, np.ndarray]:
    annotated = img.copy()
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0, 0.0, annotated
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0, 0.0, annotated
    depths = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])
        a = math.dist(start, end)
        b = math.dist(start, far)
        c = math.dist(end, far)
        # angle at 'far'
        denom = 2*b*c if (2*b*c) != 0 else 1e-6
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (b*b + c*c - a*a)/denom))))
        if angle < 90 and d > 1000:
            depths.append(d)
            cv2.circle(annotated, far, 5, (0,255,0), -1)
            cv2.line(annotated, start, far, (255,0,0), 2)
            cv2.line(annotated, end, far, (255,0,0), 2)
    fingers = min(5, len(depths) + 1 if len(depths) > 0 else 0)
    conf = min(1.0, (sum(depths) / (len(depths)+1)) / 5000.0) if depths else 0.0
    cv2.drawContours(annotated, [contour], -1, (0,255,0), 2)
    x,y,w,h = cv2.boundingRect(contour)
    cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,128,255), 2)
    return fingers, conf, annotated

def gesture_label_from_fingers(fingers: int, conf: float) -> str:
    if conf < 0.05:
        return "No hand / uncertain"
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
    return "Open hand (5+)"

def analyze_image_np(img_np: np.ndarray):
    mask, small = preprocess_hand(img_np)
    contour, area = largest_contour(mask)
    if contour is None or area < 2000:
        return {"label":"No hand","fingers":0,"confidence":0.0,"area":float(area),"annotated":img_np}
    fingers, conf, annotated = count_fingers_from_contour(contour, small)
    label = gesture_label_from_fingers(fingers, conf)
    return {"label":label,"fingers":int(fingers),"confidence":float(conf),"area":float(area),"annotated":annotated}

# UI layout
st.title("Hand Sign Detector — OpenCV (Cloud-friendly)")
st.markdown("Upload image(s) or a video to analyze. For local live webcam, see notes below.")

left, right = st.columns([2,1])
with left:
    mode = st.radio("Mode", ["Image upload","Video upload","(Local) Webcam (optional)"])
    if mode == "Image upload":
        uploaded = st.file_uploader("Upload one or more images", type=["jpg","jpeg","png"], accept_multiple_files=True)
    elif mode == "Video upload":
        uploaded = st.file_uploader("Upload a video file", type=["mp4","mov","avi"])
    else:
        uploaded = None
    process_btn = st.button("Process")
    out_area = st.empty()

with right:
    st.header("Analysis")
    last_label = st.empty()
    counters_table = st.empty()
    if "counters" not in st.session_state:
        st.session_state.counters = {}
    if "history" not in st.session_state:
        st.session_state.history = deque(maxlen=50)
    if st.button("Reset counters"):
        st.session_state.counters = {}
        st.session_state.history.clear()

def record_result(res):
    st.session_state.history.appendleft({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                         "label": res["label"],
                                         "fingers": res["fingers"],
                                         "confidence": round(res["confidence"],3),
                                         "area": round(res["area"],1)})
    st.session_state.counters[res["label"]] = st.session_state.counters.get(res["label"], 0) + 1

if process_btn:
    if mode == "Image upload":
        if not uploaded:
            st.warning("Please upload at least one image.")
        else:
            for f in uploaded:
                img_np = read_imagefile(f)
                res = analyze_image_np(img_np)
                record_result(res)
                out_area.image(res["annotated"], caption=f"{f.name} — {res['label']} (fingers={res['fingers']}, conf={res['confidence']:.2f})", use_column_width=True)
    elif mode == "Video upload":
        if not uploaded:
            st.warning("Please upload a video file.")
        else:
            video_file = uploaded
            tfile = io.BytesIO(video_file.read())
            temp_v = "temp_input.mp4"
            with open(temp_v, "wb") as fh:
                fh.write(tfile.getbuffer())
            cap = cv2.VideoCapture(temp_v)
            frames_processed = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = analyze_image_np(frame_rgb)
                if frames_processed % 10 == 0:
                    record_result(res)
                    out_area.image(res["annotated"], caption=f"Frame {frames_processed} — {res['label']}", use_column_width=True)
                frames_processed += 1
            cap.release()
            st.success(f"Processed {frames_processed} frames. See analysis on the right.")
    else:  # Local webcam
        st.warning("Local webcam mode requires 'opencv-python' installed (not headless). This mode works only when running locally.")
        # try webcam capture local
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Run locally and ensure camera is available.")
            else:
                st.info("Press the Stop button to end webcam capture.")
                stop = st.button("Stop webcam")
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = analyze_image_np(frame_rgb)
                    record_result(res)
                    out_area.image(res["annotated"], caption=f"Live — {res['label']}", use_column_width=True)
                    last_label.markdown(f"**Last:** {res['label']}  | fingers={res['fingers']} | conf={res['confidence']:.2f}")
                    counters_table.table(st.session_state.counters)
                    if stop:
                        break
                cap.release()
        except Exception as e:
            st.error("Webcam local mode error: " + str(e))

# show analysis summary
last = st.session_state.history[0] if st.session_state.history else None
if last:
    last_label.markdown(f"**Last:** {last['label']}  | fingers={last['fingers']} | conf={last['confidence']}")
counters_table.table(st.session_state.counters)
if st.session_state.history:
    st.markdown("### Recent analyses")
    st.table(list(st.session_state.history))

st.markdown("---")
st.markdown("**Notes:**")
st.markdown("- This OpenCV-based detector is a lightweight fallback suitable for hosted deployments (Spaces / Streamlit Cloud).")
st.markdown("- For best results use clear images with plain background and good lighting.")
st.markdown("- To enable local webcam mode, run locally and install `opencv-python` (not headless). On servers use the Image/Video upload modes.")
