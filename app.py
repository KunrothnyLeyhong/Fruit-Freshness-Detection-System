import os
os.environ["OPENCV_HEADLESS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="FreshCheck",
    page_icon="🍎",
    layout="centered"
)

# ---------- CUSTOM CSS (UI THEME) ----------
st.markdown("""
<style>
.title {
    font-size: 38px;
    font-weight: 700;
}
.subtitle {
    color: #4c4c4c;
}
.result-fresh {
    background: #E6F9EE;
    padding: 15px;
    border-radius: 12px;
    color: #4c4c4c;
}
.result-rotten {
    background: #FFE6E6;
    padding: 15px;
    border-radius: 12px;
    color: #4c4c4c;
</style>
""", unsafe_allow_html=True)

# ---------- LOAD LABELS ----------
with open("labels.txt", "r") as f:
    class_names = f.read().splitlines()

# ---------- LOAD MODEL ----------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.TFSMLayer(
        "model.savedmodel",
        call_endpoint="serving_default"
    )
])

# ---------- HEADER ----------
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'>🛒 FreshCheck</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-powered fruit freshness detection</p>", unsafe_allow_html=True)

# ---------- IMAGE PROCESS FUNCTION ----------
def predict_image(image: Image.Image):
    img = np.asarray(image.resize((224, 224))).astype(np.float32)
    img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    output_key = list(preds.keys())[0]
    predictions = preds[output_key].flatten()

    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx] * 100)

    return class_idx, confidence
    
# ---------- WEBCAM CLASS ----------
class FruitDetector(VideoTransformerBase):
    def __init__(self):
        self.counter = 0  # frame counter
        self.last_label = ""
        self.last_conf = 0.0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.counter += 1

        # Run prediction only every 5 frames
        if self.counter % 5 == 0:
            resized = cv2.resize(img, (224, 224))
            normalized = (resized.astype(np.float32)/127.5) - 1
            input_img = np.expand_dims(normalized, axis=0)
            preds = model.predict(input_img)
            output_key = list(preds.keys())[0]
            predictions = preds[output_key].flatten()
            class_idx = np.argmax(predictions)
            self.last_label = "FRESH" if class_idx==0 else "ROTTEN"
            self.last_conf = predictions[class_idx]*100

        # Display last prediction
        color = (0, 255, 0) if self.last_label=="FRESH" else (0, 0, 255)
        cv2.putText(
            img,
            f"{self.last_label} ({self.last_conf:.1f}%)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )
        return img

# ---------- TABS ----------
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Scan with Webcam"])

# ================== UPLOAD ==================
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload fruit image",
        type=["jpg", "jpeg", "png"]
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)

        class_idx, confidence = predict_image(image)

        if class_idx == 0:
            st.markdown(f"""
            <div class='result-fresh'>
            🟢 <b>FRESH</b><br>
            Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-rotten'>
            🔴 <b>ROTTEN</b><br>
            Confidence: {confidence:.2f}%
            </div>
            """, unsafe_allow_html=True)

# ================== WEBCAM ==================
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📷 Live Fruit Scan")
    st.caption("Point the camera at the fruit — detection runs automatically")

    webrtc_streamer(
    key="live-fruit-scan",
    video_transformer_factory=FruitDetector,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # default STUN
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
    )

    st.markdown("</div>", unsafe_allow_html=True)
# ---------- FOOTER ----------
st.markdown("---")
st.caption("Teachable Machine • TensorFlow • Streamlit")
st.caption("Developed by Kunrothny Leyhong")