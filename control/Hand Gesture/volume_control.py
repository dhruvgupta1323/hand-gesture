import cv2
import mediapipe as mp
import numpy as np
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# Audio control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]
prev_vol = minVol

# MediaPipe Hands setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.7,
                      min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

def smooth_volume(current, target, alpha=0.2):
    return (1 - alpha) * current + alpha * target

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                lmList.append((int(lm.x * w), int(lm.y * h)))

            # Thumb tip = id 4, Index fingertip = id 8
            x1, y1 = lmList[4]   # Thumb
            x2, y2 = lmList[8]   # Index

            # Draw landmarks
            cv2.circle(frame, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Distance
            dist = math.hypot(x2 - x1, y2 - y1)

            # Map distance to volume
            vol = np.interp(dist, [15, 200], [minVol, maxVol])
            smoothed_vol = smooth_volume(prev_vol, vol)
            prev_vol = smoothed_vol
            volume.SetMasterVolumeLevel(smoothed_vol, None)

            # Volume Bar
            volBar = np.interp(smoothed_vol, [minVol, maxVol], [400, 150])
            volPercent = int(np.interp(smoothed_vol, [minVol, maxVol], [0, 100]))
            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 3)
            cv2.rectangle(frame, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'{volPercent}%', (40, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe Volume Control (Thumb + Index)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
