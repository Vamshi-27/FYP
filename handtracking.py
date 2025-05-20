import cv2
import numpy as np
import pyautogui
import time
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("gesture_model.h5")
gesture_names = ["fist", "five", "none", "okay", "peace", "rad", "straight", "thumbs"]

# Webcam setup
cap = cv2.VideoCapture(0)
frame_size = 224  # Set according to your model
last_action_time = time.time()
action_delay = 1  # Delay in seconds between repeated actions

# Screen size
screen_width, screen_height = pyautogui.size()

def perform_action(gesture, cx=None, cy=None):
    global last_action_time
    now = time.time()

    # Throttle actions with delay
    if gesture in ["fist", "five", "thumbs", "rad", "straight"]:
        if now - last_action_time < action_delay:
            return
        last_action_time = now

    if gesture == "fist":
        pyautogui.click(button='left')
    elif gesture == "five":
        pyautogui.click(button='right')
    elif gesture == "thumbs":
        pyautogui.scroll(-300)  # Scroll down
    elif gesture == "rad":
        pyautogui.scroll(300)   # Scroll up
    elif gesture == "straight":
        pyautogui.hotkey('alt', 'f4')  # Close window
    elif gesture == "peace" and cx is not None and cy is not None:
        # Move mouse to a mapped location
        screen_x = np.interp(cx, [0, 640], [0, screen_width])
        screen_y = np.interp(cy, [0, 480], [0, screen_height])
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror view
    roi = frame[100:380, 100:380]  # ROI box

    # Preprocess ROI
    img = cv2.resize(roi, (frame_size, frame_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    pred_class = np.argmax(preds)
    gesture = gesture_names[pred_class]

    # Draw on frame
    cv2.rectangle(frame, (100, 100), (380, 380), (255, 0, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Use center of ROI as tracking point
    cx, cy = 240, 240
    perform_action(gesture, cx, cy)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
