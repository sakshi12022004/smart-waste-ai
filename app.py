import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image


# PAGE CONFIG 

st.set_page_config(
    page_title="Smart Waste AI",
    page_icon="♻️",
    layout="centered"
)

#  SMART CITY BACKGROUND 
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #141E30, #243B55);
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-bottom: 30px;
}

.result-card {
    background: rgba(255,255,255,0.1);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    color: white;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

#  LOAD MODEL 
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


#  HEADER 
st.markdown('<div class="title">♻️ Smart Waste Classification AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart City Waste Segregation System</div>', unsafe_allow_html=True)

#  IMAGE UPLOAD SECTION 
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

    st.markdown(f"""
    <div class="result-card">
        <h2>Prediction: {predicted_class.upper()}</h2>
        <h3>Confidence: {confidence*100:.2f}%</h3>
        <h3>Dispose in: {bin_type}</h3>
    </div>
    """, unsafe_allow_html=True)



prob_df = pd.DataFrame({
    "Category": class_labels,
    "Probability": prediction[0]
})

st.bar_chart(prob_df.set_index("Category"))

