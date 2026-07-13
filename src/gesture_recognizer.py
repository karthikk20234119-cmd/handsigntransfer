# gesture_recognizer.py
import cv2, mediapipe as mp, numpy as np, pickle, time, requests, os
from tensorflow.keras.models import load_model

SERVER_URL = "http://<SERVER_IP>:5000/api/gesture"  # replace with server IP or localhost

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

model = load_model("../models/gesture_model.h5")
with open("../models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

def extract_landmarks(landmark_list):
    return np.array([coord for lm in landmark_list for coord in (lm.x, lm.y, lm.z)], dtype=np.float32)

cap = cv2.VideoCapture(0)
last_sent = 0
SEND_INTERVAL = 1.0  # seconds between sending same predictions

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(frame_rgb)
    prediction = None
    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        X = extract_landmarks(lm).reshape(1, -1)
        probs = model.predict(X, verbose=0)[0]
        idx = np.argmax(probs)
        confidence = probs[idx]
        gesture_label = le.inverse_transform([idx])[0]
        if confidence > 0.7:
            prediction = {"gesture": gesture_label, "confidence": float(confidence)}
            now = time.time()
            if now - last_sent > SEND_INTERVAL:
                try:
                    resp = requests.post(SERVER_URL, json=prediction, timeout=1.5)
                    # optional: inspect resp.status_code
                except Exception as e:
                    print("Network error:", e)
                last_sent = now
        cv2.putText(frame, f"{gesture_label} ({confidence:.2f})", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Recognizer (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
