import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov9c.pt')
video_path = "/home/dataset/ahg2library.mp4"
cap = cv2.VideoCapture(video_path)

# print(cap.isOpened())

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv9 tracking on the frame, persisting tracks between frames

        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv9 Tracking", annotated_frame)
        key = cv2.waitKey(1)
        time.sleep(0.2)
        if key != -1 : 
            print("STOP PLAY")
            break