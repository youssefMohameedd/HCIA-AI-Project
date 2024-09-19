import torch
import cv2
import pytesseract
import pandas as pd
import time
from ultralytics import YOLO

# Load YOLO models
udacity_model = YOLO('Udacity-Cars.pt')  # YOLO model for Udacity Self-Driving Cars dataset
plate_model = YOLO('Car-Plate.pt')  # YOLO model for License Plate Recognition

# Configure Tesseract OCR path if needed (for Windows)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create a DataFrame to store plate numbers
plate_data = pd.DataFrame(columns=["Timestamp", "Plate Number"])

# Variable to track time for writing every second
last_written_time = time.time()

def extract_plate_numbers(frame, boxes):
    """
    Extract license plate numbers from detected license plates in the given frame and overlay on the frame.
    Also, write the plate numbers to an external Excel file every second.
    """
    global last_written_time, plate_data

    # Keep track of current timestamp
    current_time = time.time()
    current_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    new_entries = []  # to store new entries before writing to Excel

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

    # Check if 1 second has passed, then write to the Excel file
    if current_time - last_written_time >= 1:
        if new_entries:  # Only write if there are new plate numbers
            # Create a new DataFrame with the new entries
            new_data = pd.DataFrame(new_entries)

            # Concatenate the new data with the existing DataFrame
            plate_data = pd.concat([plate_data, new_data], ignore_index=True)
            
            # Write to Excel file (overwriting mode)
            plate_data.to_excel('Plate-Numbers.xlsx', index=False)

        last_written_time = current_time  # Reset the timer

# Path to the input video file (change to your video path)
video_path = 'first.mp4'

# Output video file name
output_video_path = 'second.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 video
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame")
        break

    # Run both YOLO models on the frame
    udacity_results = udacity_model(frame)  # Detect objects using Udacity Self-Driving Cars dataset
    plate_results = plate_model(frame)  # Detect license plates

    # Plot Udacity model results (car/object detection)
    annotated_frame = udacity_results[0].plot()

    # Process plate detection and extract numbers, display on frame
    for result in plate_results:
        extract_plate_numbers(annotated_frame, result.boxes)

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release the video capture and writer objects
cap.release()
out.release()

print(f"Video processing complete. Output saved as {output_video_path}")
