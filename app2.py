import cv2
import numpy as np
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from ultralytics import YOLO
import winsound 

# Load your trained YOLOv8 model (adjust the path if needed)
model = YOLO("C:/Users/ryfie/Downloads/RoadWatch_V1/best.pt")  # Replace with your actual model path

# Initialize Tkinter root window
root = Tk()
root.title("Road Condition Monitoring for Surface Detection")
root.geometry("1100x700")  # You can adjust the window size

# Set up the video capture (webcam)
cap = None

# Function to process the webcam frames and run inference
def process_frame():
    ret, frame = cap.read()  # Capture a frame from the webcam
    if not ret:
        print("Failed to grab frame.")
        return
    
    # Perform inference on the frame
    results = model(frame)  # Run inference on the captured frame
    annotated_frame = results[0].plot()  # Get the annotated frame with bounding boxes

    # Set confidence threshold (e.g., 0.5)
    confidence_threshold = 0.3

    
    # Check if an anomalous class is detected
    anomalous_class_detected = False
    for detection in results[0].boxes.data:
        class_id = int(detection[5])  # Assuming class ID is stored in index 5
        confidence = float(detection[4])  # Confidence score is usually in index 4
        
        # If the detection is of the anomalous class and the confidence is above the threshold
        if class_id == 0 and confidence >= confidence_threshold:  # Anomalous class ID is 0
            anomalous_class_detected = True
            break

    # If anomalous class is detected, play a beep sound
    if anomalous_class_detected:
        winsound.Beep(1000, 500)  # Frequency (1000 Hz) and duration (500 ms)

    # Convert the frame to RGB (OpenCV uses BGR by default)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to an ImageTk format
    img = Image.fromarray(annotated_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the image in the Tkinter window
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)

    # Call the function again after 10 ms to update the frame
    lbl.after(10, process_frame)

# Function to handle image selection from local storage
def select_image():
    # Open file dialog to manually select an image
    image_path = askopenfilename(
        title="Select an image", 
        initialdir="C:/Users/ryfie/Downloads/RoadWatch_V1/normal",  # Set the initial folder
        filetypes=[("Image Files", "*.jpg;*.png")]  # Allow only .jpg and .png files
    )

    if image_path:  # Check if the user selected a file
        print(f"Running inference on {image_path}")
        
        # Load the selected image
        image = cv2.imread(image_path)
        
        # Perform inference on the selected image
        results = model(image)  # Run inference on the selected image
        
        # Annotate the frame with detection results
        annotated_image = results[0].plot()  # Get the annotated frame with bounding boxes

        # Check if an anomalous class is detected in the selected image
        anomalous_class_detected = False
        for detection in results[0].boxes.data:
            class_id = int(detection[5])  # Assuming class ID is stored in index 5
            confidence = float(detection[4])  # Confidence score is in index 4
            print(f"Detected class ID: {class_id} with confidence: {confidence}")  # Print for debugging

            if class_id == 0 and confidence >= 0.3:  # Anomalous class ID is 0, and confidence threshold is 0.3
                anomalous_class_detected = True
                break
        
        # If anomalous class is detected, play a beep sound
        if anomalous_class_detected:
            winsound.Beep(1000, 500)  # Frequency (1000 Hz) and duration (500 ms)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # Convert the frame to an ImageTk format
        img = Image.fromarray(annotated_image)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the image in the Tkinter window
        lbl.imgtk = imgtk
        lbl.configure(image=imgtk)

        # Adjust the label to fit the original image size (remove any forced resizing)
        lbl.config(width=img.width, height=img.height)

    else:
        print("No image selected.")


# Function to start webcam feed
def start_webcam():
    global cap
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
    
    # Set the desired window size for webcam feed (e.g., 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)  # Set width to 1200
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720

    # Hide other options (we'll show webcam feed now)
    select_btn.pack_forget()
    webcam_btn.pack_forget()

    # Start processing the webcam frames
    process_frame()

# Define the window width and height first
window_width = 1200  # Total width of the window
window_height = 900  # Total height of the window

# Create the left frame (black background) that covers the entire left side
left_frame_width = int(window_width * 0.25)  # Left frame will take up 25% of the window width
left_frame = Frame(root, bg="black", width=left_frame_width, height=window_height)  # Full height of the window
left_frame.pack(side="left", fill="y", padx=0, pady=0)  # Fill vertically and no padding

# Load the logo image from the Downloads folder
logo_image = Image.open("C:/Users/ryfie/Downloads/RoadWatch_V1/static/roadwatch_logo.jpg")  # Path to your logo image
logo_image = logo_image.resize((400, 400), Image.Resampling.LANCZOS)  # Resize the image to fit your design
logo_photo = ImageTk.PhotoImage(logo_image)

# Create a label to display the logo image inside the left frame
logo_label = Label(left_frame, image=logo_photo, bg="black")
logo_label.pack(pady=20)  # Add padding between the logo and the buttons

# Create a frame for the buttons below the logo
button_frame = Frame(left_frame, bg="black")
button_frame.pack(pady=20)  # Add padding for the vertical space

# Create a container frame for the right section (goldenrod background)
right_frame_width = window_width - left_frame_width  # Remaining space for the right frame
right_frame = Frame(root, bg="goldenrod", width=right_frame_width, height=window_height)  # Set the height to window_height
right_frame.pack(side=RIGHT, fill="y", padx=0, pady=0)  # No padding between frames

# Create a label inside the right section to display the image or video feed
lbl = Label(right_frame)
lbl.pack(padx=20, pady=20)  # Add padding around the label

# Create a container frame for the right section (goldenrod background)
right_frame_width = window_width - left_frame_width  # Remaining space for the right frame
right_frame = Frame(root, bg="goldenrod", width=right_frame_width, height=window_height)  # Set fixed width and height
right_frame.pack(side=RIGHT, fill="both", padx=0, pady=0)  # Ensure it fills the remaining space
right_frame.pack_propagate(False)  # Prevent resizing based on content


# Create buttons for selecting image or using webcam
# Select Image Button
select_btn = Button(
    button_frame,  # Place the button inside the button_frame
    text="Select Image", 
    command=select_image, 
    bg="goldenrod",  # Button background color
    fg="black", 
    font=("Arial", 12, "bold"),  # Bold font
    width=25,  # Equal width for both buttons
    padx=5, 
    pady=5
)
select_btn.pack(pady=10)  # Increase padding between buttons

# Open Webcam Button
webcam_btn = Button(
    button_frame,  # Place the button inside the button_frame
    text="Open Webcam", 
    command=start_webcam, 
    bg="goldenrod",  # Button background color
    fg="black", 
    font=("Arial", 12, "bold"),  # Bold font
    width=25,  # Equal width for both buttons
    padx=5, 
    pady=5
)
webcam_btn.pack(pady=10)  # Increase padding between buttons

# Exit Button
exit_btn = Button(
    button_frame,  # Place the button inside the button_frame
    text="Exit", 
    command=root.quit,  # Exits the program
    bg="goldenrod",  # Button background color
    fg="black", 
    font=("Arial", 12, "bold"),  # Bold font
    width=25,  # Equal width for all buttons
    padx=5, 
    pady=5
)
exit_btn.pack(pady=10)  # Increase padding between buttons


# Run the Tkinter main loop
root.mainloop()

# Release the webcam when the app is closed
if cap:
    cap.release()
cv2.destroyAllWindows()
