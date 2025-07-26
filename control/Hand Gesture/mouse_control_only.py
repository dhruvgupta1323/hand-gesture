import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils
screenWidth, screenHeight = pyautogui.size()

cap = cv2.VideoCapture(0)

clickDown = False  # to avoid multiple clicks

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            h, w, _ = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Index & Thumb for click
            x1, y1 = lmList[8][1], lmList[8][2]  # index tip
            x2, y2 = lmList[4][1], lmList[4][2]  # thumb tip

            screenX = np.interp(x1, [0, w], [0, screenWidth])
            screenY = np.interp(y1, [0, h], [0, screenHeight])
            pyautogui.moveTo(screenX, screenY)

            # Draw circles
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 255), cv2.FILLED)

            # Distance between index and thumb
            distance = math.hypot(x2 - x1, y2 - y1)

            if distance < 30 and not clickDown:
                clickDown = True
                pyautogui.click()
                print("Click")
            elif distance >= 30:
                clickDown = False

    cv2.imshow("Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
