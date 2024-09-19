import torch
import cv2
import pytesseract
import pandas as pd
import time
import os
from ultralytics import YOLO

# Load YOLO models
udacity_model = YOLO('Udacity-Cars.pt')  # YOLO model for Udacity Self-Driving Cars dataset
plate_model = YOLO('Car-plate.pt')  # YOLO model for License Plate Recognition

# Configure Tesseract OCR path if needed (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# File paths for the CSV
csv_file = 'live_plate_numbers.csv'
temp_file = 'temp_live_plate_numbers.csv'

# Create a DataFrame to store plate numbers
plate_data = pd.DataFrame(columns=["Timestamp", "Plate Number"])

# Variable to track time for writing every second
last_written_time = time.time()

def extract_plate_numbers(frame, boxes):
    """
    Extract license plate numbers from detected license plates in the given frame and overlay on the frame.
    Also, write the plate numbers to a CSV file every second using a temporary file to avoid locking.
    """
    global last_written_time, plate_data

    # Keep track of current timestamp
    current_time = time.time()
    current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    new_entries = []  # to store new entries before writing to CSV

    for box in boxes:
        # Get bounding box coordinates for license plates
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Crop the license plate from the frame
        plate_img = frame[y1:y2, x1:x2]

        # Perform OCR to extract text from the license plate
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 8')  # psm 8 is good for single-line text
        
        # Display the extracted text on the frame
        plate_text = plate_text.strip()
        print(f"Detected License Plate Text: {plate_text}")

        # Draw bounding box for license plate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put the extracted text (plate number) on the frame
        cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Add the plate number to the list of new entries
        if plate_text:
            new_entries.append({"Timestamp": current_timestamp, "Plate Number": plate_text})

    # Check if 1 second has passed, then write to the CSV file
    if current_time - last_written_time >= 1:
        if new_entries:  # Only write if there are new plate numbers
            # Create a new DataFrame with the new entries
            new_data = pd.DataFrame(new_entries)

            # Concatenate the new data with the existing DataFrame
            plate_data = pd.concat([plate_data, new_data], ignore_index=True)
            
            # Write to a temporary CSV file first
            plate_data.to_csv(temp_file, index=False)
            
            # Rename the temp file to the actual CSV file (this avoids locking issues)
            os.replace(temp_file, csv_file)

        last_written_time = current_time  # Reset the timer

# Access the laptop camera (0 refers to the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run both YOLO models on the frame
    udacity_results = udacity_model(frame)  # Detect objects using Udacity Self-Driving Cars dataset
    plate_results = plate_model(frame)  # Detect license plates

    # Plot Udacity model results (car/object detection)
    annotated_frame = udacity_results[0].plot()

    # Process plate detection and extract numbers, display on frame
    for result in plate_results:
        extract_plate_numbers(annotated_frame, result.boxes)

    # Display the frame with both detections (car objects and license plates with text)
    cv2.imshow('YOLOv8 Live Detection with Plate Numbers', annotated_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
