import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import tempfile

st.set_page_config(page_title="Crop Disease Detection", layout="wide")

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model_path = os.path.join(os.getcwd(), "plant_disease.keras")
model = load_model(model_path)

def model_predict(image_path):
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, H, W, C)
    predictions = model.predict(img)
    result_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions) * 100.0)
    return result_index, confidence

st.header("Upload a Plant Leaf Image for Detection")
test_image = st.file_uploader("Choose an Image...", type=["jpg", "jpeg", "png"])

if test_image is not None:
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, test_image.name)
    with open(save_path, "wb") as f:
        f.write(test_image.getbuffer())

    st.image(test_image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Model is predicting..."):
            result_index, confidence = model_predict(save_path)

        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]

        if result_index < len(class_name):
            st.success(f"🌿 Predicted Disease: {class_name[result_index]}")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("Prediction index out of range. Check model/classes mapping.")
