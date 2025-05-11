import cv2
import numpy as np
from keras.models import load_model
import pyautogui

# Load model
model = load_model("model/gesture_model.h5")
classes = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']

# Webcam settings
cap = cv2.VideoCapture(0)
frame_width = 640
frame_height = 480
cap.set(3, frame_width)
cap.set(4, frame_height)

# Screen size
screen_w, screen_h = pyautogui.size()

# Define hand ROI area
box_start = (200, 100)
box_end = (440, 340)

def get_hand_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 1000:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, box_start, box_end, (255, 0, 0), 2)
    roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]

    # Preprocess for prediction
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (256, 256))
    normalized = resized.astype('float32') / 255.0
    input_tensor = np.reshape(normalized, (1, 256, 256, 1))
    pred = model.predict(input_tensor)
    label = classes[np.argmax(pred)]

    # Display prediction
    cv2.putText(frame, f"Prediction: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    if label == 'five':
        # Use binary threshold to get hand mask
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
        center = get_hand_center(thresh)
        if center:
            cx, cy = center
            # Map hand center in ROI to screen coordinates
            screen_x = np.interp(cx, [0, box_end[0] - box_start[0]], [0, screen_w])
            screen_y = np.interp(cy, [0, box_end[1] - box_start[1]], [0, screen_h])
            pyautogui.moveTo(screen_x, screen_y)
            cv2.circle(roi, (cx, cy), 5, (0, 255, 255), -1)
    elif label == 'fist':
        pyautogui.click(button='right')
    elif label == 'okay':
        pyautogui.click(button='left')
    elif label == 'peace':
        pyautogui.scroll(30)
    elif label == 'thumbs':
        pyautogui.scroll(-30)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
