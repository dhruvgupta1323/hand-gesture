import cv2
import mediapipe as mp
import pygetwindow as gw
import pyautogui
import time

# MediaPipe setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.85,
                      min_tracking_confidence=0.85)
mpDraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

gesture_time = time.time()
gesture_cooldown = 2  # seconds

def fingers_up(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    thumb_tip = 4
    thumb_ip = 3
    fingers = []

    # Thumb (horizontal check)
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_ip].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (vertical check)
    for tip_id in finger_tips:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

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
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            finger_count = fingers_up(handLms)
            now = time.time()

            if finger_count == 5 and (now - gesture_time > gesture_cooldown):
                pyautogui.hotkey('win', 'up')
                gesture_time = now
                cv2.putText(frame, 'Maximize (Palm)', (200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            elif finger_count <= 1 and (now - gesture_time > gesture_cooldown):
                pyautogui.hotkey('win', 'd')
                gesture_time = now
                cv2.putText(frame, 'Show Desktop (Fist)', (200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            elif finger_count == 2 and (now - gesture_time > gesture_cooldown):
                pyautogui.hotkey('win', 'tab')
                gesture_time = now
                cv2.putText(frame, 'Task View (2 Fingers)', (200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

            elif finger_count == 3 and (now - gesture_time > gesture_cooldown):
                pyautogui.hotkey('alt', 'tab')
                gesture_time = now
                cv2.putText(frame, 'Switch App (3 Fingers)', (200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("Hand Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
