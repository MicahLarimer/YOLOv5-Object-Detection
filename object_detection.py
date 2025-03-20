import torch  # PyTorch for loading the YOLOv5 model
import cv2    # OpenCV for handling video input and displaying output
import time   # Used to calculate FPS (Frames Per Second)

# Load YOLOv5 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise fallback to CPU
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)  # Load YOLOv5 small model

# Function to run object detection on an image
def detect_image(image_path):
    """
    Loads an image, runs YOLOv5 object detection, and displays the results.
    """
    image = cv2.imread(image_path)  # Load the image
    results = model(image)  # Run object detection
    results.render()  # Draw bounding boxes on the image

    cv2.imshow("YOLOv5 Image Detection", image)  # Display the output
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the image window

# Function to run real-time object detection using a webcam
def detect_webcam():
    """
    Captures video from the webcam, runs YOLOv5 object detection frame by frame,
    and displays FPS on the screen.
    """
    cap = cv2.VideoCapture(0)  # Open the default webcam (0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")  # Handle camera access failure
        return

    # Set webcam resolution (optional, depends on camera capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        start_time = time.time()  # Start time for FPS calculation

        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Failed to grab frame.")  # Handle failure in reading frames
            break

        results = model(frame)  # Run YOLOv5 object detection
        results.render()  # Draw bounding boxes on the frame

        # Calculate FPS (frames per second)
        fps = 20.0 / (time.time() - start_time)  # FPS = 1 / time taken for one loop
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YOLOv5 Real-Time Detection", frame)  # Show the output frame

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Choose between image or webcam detection
if __name__ == "__main__":
    mode = input("Enter 'image' to test on an image, or 'webcam' for real-time detection: ").strip().lower()
    
    if mode == "image":
        detect_image("data/sample.jpg")  # Update with the correct image path
    elif mode == "webcam":
        detect_webcam()
    else:
        print("Invalid input. Use 'image' or 'webcam'.")
