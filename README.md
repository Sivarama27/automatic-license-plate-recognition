# Automatic License Plate Recognition (ALPR)

This project implements an Automatic License Plate Recognition (ALPR) system using deep learning techniques. The system detects vehicle license plates from images and extracts the license number text automatically.

## Project Overview
The ALPR system uses a YOLOv8 model to detect license plates from vehicle images and EasyOCR to recognize the text present on the detected plates. A simple Streamlit-based web interface is provided to upload images and view the detected license plate numbers.

This project was developed as part of academic learning to gain hands-on experience in computer vision and deep learning applications.

## Technologies Used
- Python
- YOLOv8
- EasyOCR
- OpenCV
- Streamlit
- NumPy

## Features
- Detects license plates from vehicle images
- Extracts and displays license plate text
- Simple and user-friendly web interface
- End-to-end pipeline from image upload to text extraction

## Project Structure
automatic-license-plate-recognition/
│
├── app.py # Streamlit application
├── alpr.py # License plate detection and OCR logic
├── requirements.txt # Required Python libraries
├── README.md # Project documentation
└── sample_images/ # Sample input images
