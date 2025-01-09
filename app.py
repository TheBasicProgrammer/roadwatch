from flask import Flask, Response, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import base64
import os
os.environ["SDL_AUDIODRIVER"] = "dummy" 
import pygame  # Import pygame
import threading
pygame.mixer.init()

app = Flask(__name__)

exit_webcam = False
webcam_active = False  # New flag to track webcam state
webcam_lock = threading.Lock()
camera = None  # Global variable to hold the camera instance

# Initialize pygame mixer for sound playback
pygame.mixer.init()

# Load YOLO model
model = YOLO("C:/Users/ryfie/Downloads/RoadWatch_V1/best.pt")  # Update path if needed

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.3


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the uploaded image
    file = request.files['image']
    image = Image.open(file.stream)
    image_np = np.array(image)

    # Perform YOLO inference
    results = model(image_np)
    annotated_image = results[0].plot()

    # Check for anomalies
    anomalous_class_detected = False
    for detection in results[0].boxes.data:
        class_id = int(detection[5])  # Assuming class ID is stored in index 5
        confidence = float(detection[4])  # Confidence score is in index 4
        if class_id == 0 and confidence >= CONFIDENCE_THRESHOLD:
            anomalous_class_detected = True
            break

    if anomalous_class_detected:
        sound = pygame.mixer.Sound('beep.wav')  # Load the beep sound
        sound.play()  # Play the sound when anomaly detected

    # Convert annotated image back to displayable format
    _, buffer = cv2.imencode('.jpg', annotated_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"image": encoded_image, "anomaly": anomalous_class_detected})


def generate_video_feed():
    """Stream camera feed with YOLO processing."""
    global exit_webcam, camera
    try:
        camera = cv2.VideoCapture(0)  # Open camera (reinitialize each time)
        if not camera.isOpened():
            raise RuntimeError("Failed to open camera.")

        while not exit_webcam:
            ret, frame = camera.read()
            if not ret:
                break

            # Perform YOLO inference
            results = model(frame)
            annotated_frame = results[0].plot()

            # Check for anomalies
            anomalous_class_detected = False
            for detection in results[0].boxes.data:
                class_id = int(detection[5])
                confidence = float(detection[4])
                if class_id == 0 and confidence >= CONFIDENCE_THRESHOLD:
                    anomalous_class_detected = True

            # Add text overlay for anomalies
            if anomalous_class_detected:
                cv2.putText(annotated_frame, "Anomaly Detected!", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                sound = pygame.mixer.Sound('beep.wav')  # Load the beep sound
                sound.play()  # Play the sound when anomaly detected

            # Encode the frame
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_data = buffer.tobytes()

            # Yield the frame to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    finally:
        # Ensure the camera is released if an exception occurs or feed is stopped
        if camera:
            camera.release()
        camera = None


@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the camera feed."""
    global exit_webcam, webcam_active, camera
    with webcam_lock:
        if webcam_active:
            exit_webcam = True  # Signal to stop video feed
            webcam_active = False
            if camera:
                camera.release()  # Release camera resource
                camera = None
        return jsonify({"message": "Camera feed stopped successfully."})


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start the camera feed."""
    global exit_webcam, webcam_active, camera
    with webcam_lock:
        if webcam_active:
            return jsonify({"message": "Camera feed is already running."}), 400
        else:
            # Reset variables for new camera session
            exit_webcam = False
            webcam_active = True
            return jsonify({"message": "Camera feed started successfully."})


@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("Starting Flask application...")
    #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
    app.run(host="0.0.0.0", port=5000)
