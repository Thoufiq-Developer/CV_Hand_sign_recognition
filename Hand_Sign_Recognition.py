import cv2
import mediapipe as mp
import math


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)

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
    
    # We have used the distance formulaa for calculating teh distance between 2 points as we do in graphs
    distance_thumb_index = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    distance_thumb_middle = ((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2) ** 0.5

    return distance_thumb_index > 0.1 and distance_thumb_middle > 0.1

def fire_alert(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    hand_size = calculate_distance(landmarks[0], middle_tip)
    distance_middle_ring = calculate_distance(middle_tip,ring_tip)
    distance_index_middle = calculate_distance(index_tip, middle_tip)

    vulcan_salute = (
        thumb_tip.y > middle_tip.y and  
        distance_middle_ring > distance_index_middle
    )

    return vulcan_salute

def Medical_Alert(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y< thumb_tip.y and pinky_tip.y < thumb_tip.y:
        return True

def brake_fail(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    thumb_extended = thumb_tip.y < landmarks[mp_hands.HandLandmark.THUMB_IP].y
    index_extended = (
        index_tip.y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)
    mrp_curled = (
        middle_tip.y > thumb_tip.y and middle_tip.y > index_tip.y and 
        pinky_tip.y > thumb_tip.y and pinky_tip.y > index_tip.y and 
        ring_tip.y > thumb_tip.y and ring_tip.y > index_tip.y
    )
    if thumb_extended and index_extended and mrp_curled:
        return True

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if fire_alert(hand_landmarks.landmark):
                cv2.putText(frame,'Fire Signal Detected!!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif is_violence_at_home_hand(hand_landmarks.landmark):
                cv2.putText(frame, 'Hand Signal Detected!!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_hand_open(hand_landmarks.landmark):
                cv2.putText(frame, 'Help Signal Detected!! Alerting Authorities', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif Medical_Alert(hand_landmarks.landmark):
                cv2.putText(frame,'Medical Emergency Detected!! Medic Called',(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif brake_fail(hand_landmarks.landmark):
                cv2.putText(frame,'Brake Fail!! Stay Alert!!',(10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Hand Gesture Detection', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
