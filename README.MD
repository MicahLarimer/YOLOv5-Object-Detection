Overview
This project uses the pre-trained YOLOv5 model to perform real-time object detection on video frames from a webcam. The program detects various objects (e.g., people, cars, animals) and draws bounding boxes around them in the video feed.

Requirements
Python 3.x
Libraries:
torch (PyTorch)
opencv-python (OpenCV)
matplotlib (optional, for visualization)

Usage
Run the script:
python object_detection.py

The program will start the webcam and perform real-time object detection.
Detected objects will have bounding boxes around them in the video feed.
To stop the program, press the q key in the video window

Key Functions
YOLOv5 Model: Used for object detection, with the smallest model (yolov5s) for faster processing.
OpenCV (cv2): Handles webcam input and displays the detected objects in real-time.
Model Inference: Each frame captured from the webcam is passed through the YOLOv5 model to detect objects.

Code Summary
Imports:
torch: For loading YOLOv5.
cv2: For video capture and displaying the output.
Webcam Setup: Captures video frames from the webcam.
YOLOv5 Inference: Detects objects in each frame.
Rendering: Draws bounding boxes around detected objects.
Display: Shows the frames with detections in real-time using cv2.imshow().