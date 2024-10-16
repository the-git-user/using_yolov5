import cv2
import torch

# Load the trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model(frame)

    # Display results
    cv2.imshow('Uno Card Detection', results.render()[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
