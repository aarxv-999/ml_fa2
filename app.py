import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="PPE Detection Dashboard", page_icon="🦺", layout="centered")

st.title("🦺 PPE Detection Dashboard")
st.write("Upload an image to check worker safety compliance using the trained YOLOv8 model.")

# Load your trained model
model = YOLO("best.pt")  # keep best.pt in same folder as this file

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting PPE items..."):
        results = model.predict(source=np.array(img), conf=0.25, device="cpu")

    res = results[0]
    img_with_boxes = res.plot()
    st.image(img_with_boxes, caption="Detection Results", use_column_width=True)

    # Compliance logic
    detected = [res.names[int(i)] for i in res.boxes.cls]

    if all(x in detected for x in ["Hardhat", "Mask", "Safety Vest"]):
        st.success("✅ Fully Compliant (Green)")
    elif any(x in detected for x in ["Hardhat", "Mask", "Safety Vest"]):
        st.warning("🟡 Partially Compliant (Yellow)")
    else:
        st.error("🚫 Non-Compliant (Red)")
