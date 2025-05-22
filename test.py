import cv2
import numpy as np
import pyautogui
import tensorflow as tf
import time
import mediapipe as mp

# Load gesture model
model = tf.keras.models.load_model("model/gesture_model.h5")
class_names = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']

# Webcam
cap = cv2.VideoCapture(0)

# Screen size for pyautogui
screen_w, screen_h = pyautogui.size()
prev_x, prev_y = 0, 0

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Preprocess image for model
def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray, (256, 256))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)      # Batch dimension
    img = np.expand_dims(img, axis=-1)     # Channel dimension
    return img

# Predict gesture
def predict_gesture(frame):
    preprocessed = preprocess(frame)
    predictions = model.predict(preprocessed, verbose=0)
    index = np.argmax(predictions)
    return class_names[index]

print("[INFO] Starting gesture control. Press 'q' to quit.")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_roi)
    gesture = "none"

    if result.multi_hand_landmarks is not None:
        gesture = predict_gesture(roi)
        print(f"[INFO] Predicted gesture: {gesture}")

        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === Real gesture-based actions ===
        if gesture == "peace":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    screen_x = int(screen_w * (cx / 300))
                    screen_y = int(screen_h * (cy / 300))
                    smooth_x = prev_x + (screen_x - prev_x) // 5
                    smooth_y = prev_y + (screen_y - prev_y) // 5
                    pyautogui.moveTo(smooth_x, smooth_y)
                    prev_x, prev_y = smooth_x, smooth_y
                    print(f"[ACTION] Move mouse to: ({smooth_x}, {smooth_y})")

        elif gesture == "fist":
            pyautogui.click(button="left")
            print("[ACTION] Left click")
            time.sleep(0.5)

        elif gesture == "five":
            pyautogui.click(button="right")
            print("[ACTION] Right click")
            time.sleep(0.5)

        elif gesture == "thumbs":
            pyautogui.scroll(-20)
            print("[ACTION] Scroll down")
            time.sleep(0.3)

        elif gesture == "rad":
            pyautogui.scroll(20)
            print("[ACTION] Scroll up")
            time.sleep(0.3)

        elif gesture == "straight":
            pyautogui.hotkey('alt', 'f4')
            print("[ACTION] Close window")
            time.sleep(1)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Draw ROI
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Show video
    cv2.imshow("Hand Gesture Control", frame)
    cv2.imshow("ROI", roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
