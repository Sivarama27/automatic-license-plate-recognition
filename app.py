import streamlit as st
from ultralytics import YOLO
import cv2
import easyocr
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

# ----------------- Setup -----------------
st.set_page_config(page_title="ALPR System", layout="centered")
st.title("🚗 Automatic License Plate Recognition (ALPR)")

# Create folders if not exist
os.makedirs("cropped_plates", exist_ok=True)

# Load model & OCR
model = YOLO("models/best.pt")
reader = easyocr.Reader(['en'])

# CSV file
CSV_FILE = "results.csv"

# ----------------- Helper Functions -----------------
def clean_plate_text(text_list):
    raw_text = " ".join(text_list).upper()
    raw_text = re.sub(r'[^A-Z0-9]', '', raw_text)

    corrections = {
        'O': '0',
        'I': '1',
        'L': '1',
        'B': '8',
        'S': '5'
    }

    final_text = ""
    for ch in raw_text:
        final_text += corrections.get(ch, ch)

    return final_text


def save_to_csv(data):
    df = pd.DataFrame([data])
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, index=False)


# ----------------- Streamlit UI -----------------
uploaded_file = st.file_uploader("📤 Upload a vehicle image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", channels="BGR")

    # Run YOLO
    results = model(img)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop license plate
            plate_img = img[y1:y2, x1:x2]

            # Show cropped plate
            st.subheader("🔍 Cropped License Plate")
            st.image(plate_img, channels="BGR")

            # OCR
            ocr_text = reader.readtext(plate_img, detail=0)
            final_plate = clean_plate_text(ocr_text)

            st.subheader("📄 Recognized Plate Number")
            st.success(final_plate if final_plate else "Not detected")

            # Save cropped image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plate_img_name = f"cropped_plates/plate_{timestamp}.jpg"
            cv2.imwrite(plate_img_name, plate_img)

            # Save CSV
            data = {
                "timestamp": timestamp,
                "plate_number": final_plate,
                "image_path": plate_img_name
            }
            save_to_csv(data)

            st.info("✅ Saved cropped image and CSV entry")
