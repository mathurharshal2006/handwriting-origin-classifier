
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="Handwriting Origin Identifier",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwriting Origin Identifier")
st.markdown("### Upload a handwriting image to identify its origin!")
st.markdown("---")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="model_fixed.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error("Model error: " + str(e))
    interpreter = None

COUNTRIES = ["🇮🇳 Indian","🇺🇸 American",
             "🇨🇳 Chinese","🇯🇵 Japanese"]
COLORS    = ["#FF9933","#3C3B6E",
             "#DE2910","#BC002D"]

def predict(image, interpreter):
    img = image.convert("L").resize((128,128))
    img_array = np.array(img,
                dtype=np.float32)/255.0
    img_array = img_array.reshape(1,128,128,1)
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(
        inp[0]["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(
        out[0]["index"])[0]

if interpreter is not None:
    st.markdown("### Upload Handwriting Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg","jpeg","png"]
    )
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        with col1:
            st.markdown("#### Your Image")
            st.image(image, use_column_width=True)
        with st.spinner("Analyzing handwriting..."):
            preds      = predict(image, interpreter)
            pred_class = np.argmax(preds)
            confidence = preds[pred_class] * 100
        with col2:
            st.markdown("#### Result")
            st.markdown(
                f"<h2 style=color:{COLORS[pred_class]}>"
                f"{COUNTRIES[pred_class]}</h2>",
                unsafe_allow_html=True)
            st.metric("Confidence",
                      f"{confidence:.1f}%")
        st.markdown("---")
        st.markdown("#### Confidence For Each Country")
        for i in range(4):
            prob = float(preds[i]) * 100
            st.markdown(f"**{COUNTRIES[i]}**")
            st.progress(int(prob))
            st.caption(f"{prob:.1f}%")
        st.markdown("---")
        facts = {
            0: "Indian handwriting tends to be upright with round uniform letters!",
            1: "American handwriting often shows a right-leaning slant!",
            2: "Chinese writers bring brush-stroke precision to English!",
            3: "Japanese handwriting is extremely consistent in size!"
        }
        st.info(facts[int(pred_class)])

st.markdown("---")
st.markdown(
    "Built by Harshal Mathur | "
    "CNN Model | 70.63% Accuracy | "
    "Beats research (53%)"
)
