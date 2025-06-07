import streamlit as st
import numpy as np
import pickle
from PIL import Image, ImageOps

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Digit Recognizer", layout="centered")

# itle
st.markdown("# Digit Recognition Web App")
st.markdown("Upload an image of a handwritten digit (0–9) and get the prediction!")

# Upload image
uploaded_file = st.file_uploader("Upload an image (8x8 pixel)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image_resized = image.resize((8, 8), Image.Resampling.LANCZOS)
    image_resized = ImageOps.invert(image_resized)  # Invert to match MNIST style
    image_array = np.array(image_resized) / 16.0     # Scale down values
    flat_data = image_array.reshape(1, -1)

    # Predict
    prediction = model.predict(flat_data)[0]
    st.success(f"Predicted Digit: *{prediction}*")