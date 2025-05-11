import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("model/gesture_model.h5")
classes = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']  # Order matters

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (256, 256))
    roi = roi.astype('float32') / 255.0
    roi = np.reshape(roi, (1, 256, 256, 1))

    pred = model.predict(roi)
    label = classes[np.argmax(pred)]

    cv2.putText(frame, f"Prediction: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
