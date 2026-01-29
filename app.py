import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------- PAGE CONFIG --------
st.set_page_config(
    page_title="FreshCheck 🛒",
    page_icon="🍎",
    layout="centered"
)

# -------- LOAD LABELS --------
with open("labels.txt", "r") as f:
    class_names = f.read().splitlines()

# -------- LOAD MODEL (Python 3.13 + SavedModel via TFSMLayer) --------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.TFSMLayer(
        "model.savedmodel",
        call_endpoint="serving_default"
    )
])

# -------- HEADER --------
st.markdown("<h1 style='text-align:center;'>🛒 FreshCheck</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>AI fruit freshness detection for supermarkets</p>", unsafe_allow_html=True)

# -------- UPLOAD & PREDICTION --------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload fruit image", 
    type=["jpg", "jpeg", "png"], 
    key="fruit_image_uploader"  # unique key
)
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded fruit", use_column_width=True)

    # Preprocess (MobileNet TM scaling example)
    img = np.asarray(image.resize((224, 224))).astype(np.float32)
    img = (img / 127.5) - 1
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)  # dict with keys like 'fresh_fruit (apple, banana, strawberry)'
    
    # Extract predictions for each class
    output_key = list(preds.keys())[0]  # Get the output layer key
    predictions = np.array(preds[output_key]).flatten()
    
    # Get class index with highest confidence
    class_idx = np.argmax(predictions)
    confidence = float(predictions[class_idx] * 100)
    
    # Display based on class index
    if class_idx == 0:  # Fresh fruit class
        st.success("🟢 **Fruit is FRESH**")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.caption("✅ Safe to display on supermarket shelves")
    else:  # Rotten fruit class
        st.error("🔴 **Fruit is ROTTEN**")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.caption("❌ Remove from shelf immediately")



# -------- FOOTER --------
st.markdown("---")
st.caption("Teachable Machine • TensorFlow • Streamlit")
