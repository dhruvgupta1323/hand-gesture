# import cv2
# import numpy as np
# import tensorflow as tf

# # Load trained model
# model = tf.keras.models.load_model('model/hand_gesture_model.h5')

# # Class labels used during training (must match training order)
# class_names = ['Palm', 'Fist']

# # Map gestures to light status
# gesture_to_action = {
#     'Palm': 'ON',
#     'Fist': 'OFF'
# }

# # Load bulb images
# bulb_on = cv2.imread('bulb_on.png')
# bulb_off = cv2.imread('bulb_off.png')

# # Check image loading
# if bulb_on is None or bulb_off is None:
#     print("❌ ERROR: bulb_on.png or bulb_off.png not found or failed to load.")
#     exit()

# # Resize bulb images for display box
# bulb_w, bulb_h = 200, 200
# bulb_on = cv2.resize(bulb_on, (bulb_w, bulb_h))
# bulb_off = cv2.resize(bulb_off, (bulb_w, bulb_h))

# # Start webcam
# cap = cv2.VideoCapture(0)
# light_status = 'OFF'

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Region of interest (where to show hand gesture)
#     x1, y1, x2, y2 = 100, 100, 300, 300
#     roi = frame[y1:y2, x1:x2]
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Preprocess image for model
#     roi_resized = cv2.resize(roi, (64, 64))
#     roi_normalized = roi_resized.astype("float32") / 255.0
#     roi_input = np.expand_dims(roi_normalized, axis=0)

#     # Predict the gesture
#     prediction = model.predict(roi_input)
#     predicted_class = np.argmax(prediction)
#     label = class_names[predicted_class]
#     confidence = prediction[0][predicted_class] * 100

#     # Update light status if valid gesture
#     if label in gesture_to_action:
#         light_status = gesture_to_action[label]

#     # Choose bulb image based on light status
#     bulb_img = bulb_on if light_status == 'ON' else bulb_off

#     # Position to display bulb (top-right corner)
#     bulb_x, bulb_y = 420, 40
#     frame[bulb_y:bulb_y+bulb_h, bulb_x:bulb_x+bulb_w] = bulb_img
#     cv2.rectangle(frame, (bulb_x, bulb_y), (bulb_x+bulb_w, bulb_y+bulb_h), (255, 255, 255), 2)

#     # Display label and light status
#     display_text = f"{label} ({confidence:.1f}%) | Light: {light_status}"
#     cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
#     cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

#     # Show final output
#     cv2.imshow("Gesture-Controlled Light", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import cv2
# import numpy as np
# import tensorflow as tf

# # Load the trained model
# model = tf.keras.models.load_model('model/hand_gesture_model.h5')

# # Class labels (must match training order)
# class_names = ['Palm', 'Fist']

# # Gesture to action mapping
# gesture_to_action = {
#     'Palm': 'ON',
#     'Fist': 'OFF'
# }

# # Load and resize bulb images
# bulb_on = cv2.imread('bulb_on.png')
# bulb_off = cv2.imread('bulb_off.png')

# if bulb_on is None or bulb_off is None:
#     print("❌ ERROR: bulb_on.png or bulb_off.png not found.")
#     exit()

# bulb_w, bulb_h = 200, 200
# bulb_on = cv2.resize(bulb_on, (bulb_w, bulb_h))
# bulb_off = cv2.resize(bulb_off, (bulb_w, bulb_h))

# # Webcam
# cap = cv2.VideoCapture(0)
# light_status = 'OFF'

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Define ROI for hand
#     x1, y1, x2, y2 = 100, 100, 300, 300
#     roi = frame[y1:y2, x1:x2]
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Preprocess for model
#     roi_resized = cv2.resize(roi, (64, 64))
#     roi_normalized = roi_resized.astype("float32") / 255.0
#     roi_input = np.expand_dims(roi_normalized, axis=0)

#     # Predict gesture
#     prediction = model.predict(roi_input)
#     predicted_class = np.argmax(prediction)

#     if predicted_class < len(class_names):
#         label = class_names[predicted_class]
#         confidence = prediction[0][predicted_class] * 100

#         # Update light status
#         if label in gesture_to_action:
#             light_status = gesture_to_action[label]

#         display_text = f"{label} ({confidence:.1f}%) | Light: {light_status}"
#     else:
#         label = "Unknown"
#         confidence = 0.0
#         display_text = "Unrecognized gesture"

#     # Show bulb
#     bulb_img = bulb_on if light_status == 'ON' else bulb_off
#     bulb_x, bulb_y = 420, 40
#     frame[bulb_y:bulb_y+bulb_h, bulb_x:bulb_x+bulb_w] = bulb_img
#     cv2.rectangle(frame, (bulb_x, bulb_y), (bulb_x+bulb_w, bulb_y+bulb_h), (255, 255, 255), 2)

#     # Show gesture info
#     cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 4)
#     cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

#     # Show result
#     cv2.imshow("Gesture-Controlled Light", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()






# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------





# import cv2
# import mediapipe as mp
# import numpy as np

# # Load and resize light bulb images
# bulb_on = cv2.resize(cv2.imread("bulb_on.png"), (200, 200))
# bulb_off = cv2.resize(cv2.imread("bulb_off.png"), (200, 200))

# # MediaPipe hand detection setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False,
#                        max_num_hands=1,
#                        min_detection_confidence=0.85,   
#                        min_tracking_confidence=0.85)
# mp_draw = mp.solutions.drawing_utils

# # Webcam setup
# cap = cv2.VideoCapture(0)
# light_status = 'OFF'

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)  # Mirror the image (like a selfie camera)
#     h, w, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     label = "No Hand"

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             lm = handLms.landmark

#             # Finger state detection using landmark positions
#             fingers = []
#             fingers.append(lm[8].y < lm[6].y)   # Index finger
#             fingers.append(lm[12].y < lm[10].y) # Middle finger
#             fingers.append(lm[16].y < lm[14].y) # Ring finger
#             fingers.append(lm[20].y < lm[18].y) # Pinky

#             extended = fingers.count(True)

#             if extended >= 3:
#                 label = "Palm"
#                 light_status = "ON"
#             else:
#                 label = "Fist"
#                 light_status = "OFF"

#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#     # Display bulb based on light status
#     bulb = bulb_on if light_status == "ON" else bulb_off
#     frame[40:240, 420:620] = bulb
#     cv2.rectangle(frame, (420, 40), (620, 240), (255, 255, 255), 2)

#     # Display gesture and status text
#     text = f"{label} | Light: {light_status}"
#     cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9, (0, 0, 0), 4)
#     cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9, (0, 255, 255), 2)

#         # Show output in full screen
#     cv2.namedWindow("Gesture Light Control", cv2.WND_PROP_FULLSCREEN)
#     cv2.setWindowProperty("Gesture Light Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     cv2.imshow("Gesture Light Control", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




#  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import cv2
import mediapipe as mp
import numpy as np
import serial
import time

# ✅ Initialize Serial Communication with Arduino
try:
    ser = serial.Serial('COM5', 9600)  # Make sure COM5 matches your board
    time.sleep(2)
    print("✅ Connected to COM5")
except serial.SerialException as e:
    print(f"❌ Could not open COM5: {e}")
    exit(1)

# ✅ Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.85,
                       min_tracking_confidence=0.85)
mp_draw = mp.solutions.drawing_utils

# ✅ Start Camera
cap = cv2.VideoCapture(0)
light_status = 'OFF'
gesture_label = 'No Hand'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_label = "No Hand"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark

            # ✅ Detect Finger States (Index, Middle, Ring, Pinky)
            fingers = []
            fingers.append(lm[8].y < lm[6].y)    # Index
            fingers.append(lm[12].y < lm[10].y)  # Middle
            fingers.append(lm[16].y < lm[14].y)  # Ring
            fingers.append(lm[20].y < lm[18].y)  # Pinky

            extended = fingers.count(True)

            # ✅ Determine Gesture
            if extended >= 3:
                gesture_label = "Palm"
                if light_status != "ON":
                    light_status = "ON"
                    # print("✅ Gesture: Palm detected → Turning Light ON")
                    ser.write(b'ON\n')
            else:
                gesture_label = "Fist"
                if light_status != "OFF":
                    light_status = "OFF"
                    # print("✅ Gesture: Fist detected → Turning Light OFF")
                    ser.write(b'OFF\n')

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # ❌ Removed Bulb Image Drawing Section

    # ✅ Display Status Text
    status_text = f"{gesture_label} | Light: {light_status}"
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 4)
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2)

    # ✅ Show Fullscreen Preview
    cv2.namedWindow("Gesture Light Control", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Gesture Light Control", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Gesture Light Control", frame)

    # ✅ Break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
ser.close()

