# capture_dataset.py
import cv2, mediapipe as mp
import numpy as np
import pandas as pd
import os, time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

OUT_DIR = "../dataset"
os.makedirs(OUT_DIR, exist_ok=True)

def landmarks_to_list(landmarks):
    return [coord for lm in landmarks for coord in (lm.x, lm.y, lm.z)]

def main():
    cap = cv2.VideoCapture(0)
    gesture_name = input("Enter gesture name (single word, e.g., Hello): ").strip()
    samples = int(input("How many samples to capture (e.g., 200): "))
    rows = []
    count = 0
    print("Press SPACE to capture a frame for label or ESC to quit.")
    while cap.isOpened() and count < samples:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Captured: {count}/{samples}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Capture Dataset (Press SPACE to save)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                row = landmarks_to_list([(l.x, l.y, l.z) for l in lm])
                row = [item for sub in lm for item in (sub.x, sub.y, sub.z)]  # alternative format
                rows.append(row + [gesture_name])
                count += 1
            else:
                print("No hand detected; try again.")
        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    if rows:
        df = pd.DataFrame(rows)
        timestamp = int(time.time())
        out_file = os.path.join(OUT_DIR, f"{gesture_name}_{timestamp}.csv")
        df.to_csv(out_file, index=False)
        print("Saved dataset to", out_file)
    else:
        print("No data captured.")

if __name__ == "__main__":
    main()
