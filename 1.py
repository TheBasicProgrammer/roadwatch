import os
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from ultralytics import YOLO

# Initialize Tkinter and hide the root window
Tk().withdraw()

# Open file dialog to manually select an image
image_path = askopenfilename(
    title="Select an image", 
    initialdir="C:/Users/Catherina Dela Cruz/Desktop",  # Set the initial folder
    filetypes=[("Image Files", "*.jpg;*.png")]  # Allow only .jpg and .png files
)

if image_path:  # Check if the user selected a file
    print(f"Running inference on {image_path}")
    
    # Load your trained YOLOv8 model (adjust the path if needed)
    model = YOLO("C:/Users/Catherina Dela Cruz/Desktop/RoadWatch_V1/best.pt")  # Replace with your actual model path

    # Perform inference on the selected image
    results = model(image_path)  # Directly use the selected image path
    
    # Since 'results' is a list, take the first result
    result = results[0]  # Get the first result

    # Show results (bounding boxes and labels)
    result.show()  # This should now work

    # Optionally, print detailed results (e.g., bounding boxes, labels)
    print(result.pandas().xywh)  # Print bounding box details: x, y, width, height format
else:
    print("No image selected. Exiting...")
