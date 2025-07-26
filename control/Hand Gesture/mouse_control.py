import cv2
import mediapipe as mp
import pyautogui
import numpy as np  # ✅ Add this line
import time

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Screen size
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0
smoothening = 5

# FPS Tracker
prev_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Mirror the image (makes it natural to user)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        lmList = []
        handLms = results.multi_hand_landmarks[0]
        for id, lm in enumerate(handLms.landmark):
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append((id, cx, cy))

        if len(lmList) >= 9:
            x1, y1 = lmList[8][1], lmList[8][2]
            # Map hand coordinates to screen size
            screen_x = int(np.interp(x1, [0, w], [0, screen_w]))
            screen_y = int(np.interp(y1, [0, h], [0, screen_h]))
            cur_x = prev_x + (screen_x - prev_x) // smoothening
            cur_y = prev_y + (screen_y - prev_y) // smoothening
            pyautogui.moveTo(cur_x, cur_y)
            prev_x, prev_y = cur_x, cur_y

        if len(lmList) >= 12:
            ix, iy = lmList[8][1], lmList[8][2]
            mx, my = lmList[12][1], lmList[12][2]
            distance = ((ix - mx)**2 + (iy - my)**2)**0.5
            if distance < 35:
                pyautogui.click()
                time.sleep(0.25)

    # Show FPS
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Natural Hand Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
