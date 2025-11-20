import os
import streamlit as st
import cv2
import numpy as np
import tempfile
from io import BytesIO
import torch
from ultralytics import YOLO
import requests

# ===============================
# Force CPU-only mode (safe for Windows / Streamlit Cloud)
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
os.environ["FORCE_CPU"] = "1"

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(page_title="Vehicle Image Recognition", layout="wide")
st.title("Vehicle Image Recognition System")
st.markdown(
    "Upload a vehicle image below to **detect and crop license plates**. "
    "This demo uses a fine-tuned YOLO model."
)

# Sidebar config
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.2, 0.9, 0.3, 0.05)

# ===============================
# Model Download (Streamlit-friendly)
# ===============================
MODEL_URL = "https://drive.google.com/uc?export=download&id=1fvmgFAWSb4Yw-jARzdZOueDNllOS14W-"
MODEL_PATH = "best_plate_finetune.pt"

@st.cache_resource
def load_model(model_path):
    if not os.path.exists(model_path):
        st.info("Downloading YOLO model...")
        r = requests.get(MODEL_URL, stream=True)
        with open(model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded!")
    return YOLO(model_path)

model = load_model(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"**Model Running On:** {device.upper()}")

# ===============================
# File Upload & Detection
# ===============================
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    temp_dir = tempfile.gettempdir()
    img_path = os.path.join(temp_dir, f"uploaded_image{file_ext}")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    img = cv2.imread(img_path)
    if img is None:
        st.error("❌ Could not read uploaded image.")
        st.stop()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with st.spinner("🔍 Detecting license plate..."):
        results = model.predict(source=img_path, conf=conf_threshold, device=device, verbose=False)

    cropped_plates = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = np.clip([x1, y1, x2, y2],
                                 0, [img.shape[1]-1, img.shape[0]-1, img.shape[1]-1, img.shape[0]-1])
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 120), 3)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size > 0:
            cropped_plates.append((conf, crop))

    if cropped_plates:
        cropped_plates.sort(key=lambda x: x[0], reverse=True)
        best_conf, best_crop = cropped_plates[0]
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.image(img_rgb, caption="Detected Vehicle", use_container_width=True)
        with col2:
            st.image(best_crop, caption=f"Detected Plate (Conf: {best_conf:.2f})", use_container_width=True)
            _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(best_crop, cv2.COLOR_RGB2BGR))
            st.download_button(
                "📥 Download Cropped Plate",
                data=BytesIO(img_encoded.tobytes()),
                file_name="cropped_plate.jpg",
                mime="image/jpeg"
            )
    else:
        st.warning("⚠️ No license plate detected. Please try a clearer image.")
        st.image(img_rgb, caption="Vehicle Image", use_container_width=True)

# ===============================
# Device Status Badge
# ===============================
st.markdown(
    f"<div style='padding:10px;border-radius:8px;"
    f"background-color:#{'00ff88' if device=='cpu' else '88ccff'};"
    f"color:#000;font-weight:bold;'>"
    f"Device in use: {device.upper()}"
    f"</div>", unsafe_allow_html=True
)
