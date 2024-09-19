## HCIA-AI-Project
This is Git Repository of my Final project of HCIA-AI Huawei Course

**# AI-powered Cityscape Camera**:
Final Project for AI Course by the National Telecommunications Institute in partnership with Huawei Egyptian Talents Academy (ETA)
# Project Overview
The AI-powered Cityscape Camera is an object detection system designed to identify and analyze common objects in urban environments using YOLOv8, as well as recognize car plate numbers via Optical Character Recognition (OCR). The project leverages a deep learning model trained on the COCO dataset and a Cityscape dataset, which enhances object detection performance in real-time camera feeds.

This project is particularly useful in ADAS Systems, traffic surveillance for automatic plate recognition, smart city security systems, and garage entry monitoring.

## Features:
Object Detection: Detection of cityscape objects such as pedestrians, cars, trucks, bikers, and traffic lights.
Car Plate Recognition: Uses the Tesseract OCR library to recognize and extract car plate numbers from camera streams.
Data Export: Captured data (plate numbers and corresponding timestamps) are exported to a CSV file for further analysis.
## Technologies Used:
YOLOv8: Pretrained YOLOv8n model on COCO Dataset, fine-tuned for cityscape objects.
Tesseract OCR: Used to recognize and extract text from car plates.
Pandas: For CSV data manipulation and export.
OpenCV: For live camera stream handling and object detection visualization.
Python: Primary programming language for model implementation and data processing.
## Project Workflow:
Data Acquisition: The system collects live video feed data via a connected camera.
Data Processing: The deep learning model processes the feed, detecting objects and identifying car plates in real-time.
Data Analysis: Detected plate numbers are timestamped and exported to a CSV file for tracking.
