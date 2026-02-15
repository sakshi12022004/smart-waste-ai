import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="centered"
)

# Load Model
@st.cache_resource
def load_trained_model():
    return load_model("smart_waste_classifier_mobilenet.h5")

model = load_trained_model()

class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def get_bin_recommendation(label):
    if label in ["paper", "cardboard"]:
        return "🟢 Green Bin (Dry Waste)"
    elif label in ["plastic", "metal"]:
        return "🟡 Yellow Bin (Recyclable Waste)"
    elif label == "glass":
        return "⚪ White Bin (Glass Waste)"
    else:
        return "⚫ Black Bin (General Waste)"

st.title("♻️ Smart Waste Classification AI")
st.subheader("Smart City Waste Segregation System")

# ================= UPLOAD SECTION =================
st.markdown("## 📤 Upload Waste Image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=400)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    predicted_class = class_labels[np.argmax(prediction)]
    bin_type = get_bin_recommendation(predicted_class)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence*100:.2f}%")
    st.write(f"Dispose in: {bin_type}")

    prob_df = pd.DataFrame({
        "Category": class_labels,
        "Probability": prediction[0]
    })

    st.bar_chart(prob_df.set_index("Category"))

# ================= CAMERA SECTION =================
st.markdown("## 📷 Capture Waste Image")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:
    img = Image.open(camera_image)
    st.image(img, caption="Captured Image", width=400)

    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    confidence = float(np.max(prediction))
    predicted_class = class_labels[np.argmax(prediction)]
    bin_type = get_bin_recommendation(predicted_class)

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence*100:.2f}%")
    st.write(f"Dispose in: {bin_type}")

    prob_df = pd.DataFrame({
        "Category": class_labels,
        "Probability": prediction[0]
    })

    st.bar_chart(prob_df.set_index("Category"))
