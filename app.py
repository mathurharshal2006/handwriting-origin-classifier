import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Handwriting Origin Identifier",
    page_icon="✍️",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("✍️ Handwriting Origin Identifier")
st.markdown("### Identify nationality from English handwriting!")
st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_path = "final_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

# ---------------- DATA ----------------
COUNTRIES = ["Indian", "American", "Chinese", "Japanese"]
FLAGS     = ["🇮🇳", "🇺🇸", "🇨🇳", "🇯🇵"]
COLORS    = ["#FF9933", "#3C3B6E", "#DE2910", "#BC002D"]

FACTS = [
    "Indian handwriting is upright with round uniform letters!",
    "American handwriting shows right-leaning cursive influence!",
    "Chinese writers bring brush-stroke precision to English!",
    "Japanese handwriting is extremely consistent in sizing!"
]

# ---------------- CHECK MODEL ----------------
if model is None:
    st.error("❌ Model not found! Make sure 'final_model.h5' is in your repo.")
else:
    st.success("✅ AI Model loaded successfully!")

    # ---------------- UPLOAD ----------------
    st.markdown("### 📸 Upload Handwriting Image")
    uploaded_file = st.file_uploader(
        "Choose a clear handwriting image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        # ---------------- SHOW IMAGE ----------------
        with col1:
            st.markdown("#### Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        # ---------------- PREPROCESS ----------------
        img = image.convert("L")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 1)

        # ---------------- PREDICT ----------------
        with st.spinner("🔍 Analyzing handwriting..."):
            predictions = model.predict(img_array, verbose=0)
            pred_class = np.argmax(predictions[0])
            confidence = predictions[0][pred_class] * 100

        # ---------------- RESULT ----------------
        with col2:
            st.markdown("#### Result")
            st.markdown(
                f"<h1 style='text-align:center'>{FLAGS[pred_class]}</h1>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<h3 style='text-align:center; color:{COLORS[pred_class]}'>"
                f"{COUNTRIES[pred_class]} Origin</h3>",
                unsafe_allow_html=True
            )
            st.metric("Confidence", f"{confidence:.1f}%")

        # ---------------- ALL PROBABILITIES ----------------
        st.markdown("---")
        st.markdown("#### 📊 All Country Scores")

        for i in range(4):
            prob = predictions[0][i] * 100
            st.progress(
                int(prob),
                text=f"{FLAGS[i]} {COUNTRIES[i]} — {prob:.1f}%"
            )

        # ---------------- FUN FACT ----------------
        st.markdown("---")
        st.info(f"💡 {FACTS[pred_class]}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray'>"
    "Built by Harshal Mathur | CNN Model | "
    "Handwriting Origin Classifier</p>",
    unsafe_allow_html=True
)
