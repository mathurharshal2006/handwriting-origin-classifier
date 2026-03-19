

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Handwriting Origin Identifier",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwriting Origin Identifier")
st.markdown("### Identify nationality from English handwriting!")
st.markdown("---")

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    st.success("✅ AI Model loaded!")
except Exception as e:
    st.error("Model error: " + str(e))
    interpreter = None

COUNTRIES = ["Indian","American","Chinese","Japanese"]
FLAGS     = ["🇮🇳","🇺🇸","🇨🇳","🇯🇵"]
COLORS    = ["#FF9933","#3C3B6E","#DE2910","#BC002D"]
FACTS = [
    "Indian handwriting is upright with round uniform letters!",
    "American handwriting shows right-leaning cursive influence!",
    "Chinese writers bring brush-stroke precision to English!",
    "Japanese handwriting is extremely consistent in sizing!"
]

def predict(image, interpreter):
    img = image.convert("L").resize((128,128))
    img_array = np.array(img, dtype=np.float32)/255.0
    img_array = img_array.reshape(1,128,128,1)
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(inp[0]["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(out[0]["index"])[0]

if interpreter is not None:
    st.markdown("### 📸 Upload Handwriting Image")
    uploaded_file = st.file_uploader(
        "Choose a clear handwriting photo",
        type=["jpg","jpeg","png"]
    )
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        with col1:
            st.markdown("#### Your Image")
            st.image(image, use_column_width=True)
        with st.spinner("🔍 Analyzing..."):
            preds      = predict(image, interpreter)
            pred_class = np.argmax(preds)
            confidence = preds[pred_class] * 100
        with col2:
            st.markdown("#### Result")
            st.markdown(
                "<h1 style=text-align:center>" +
                FLAGS[pred_class] + "</h1>",
                unsafe_allow_html=True)
            st.markdown(
                "<h3 style=text-align:center;color:" +
                COLORS[pred_class] + ">" +
                COUNTRIES[pred_class] +
                " Origin</h3>",
                unsafe_allow_html=True)
            st.metric("Confidence",
                      f"{confidence:.1f}%")
        st.markdown("---")
        st.markdown("#### 📊 All Country Scores")
        for i in range(4):
            prob = float(preds[i]) * 100
            st.progress(int(prob),
                text=FLAGS[i]+" "+COUNTRIES[i]+
                " — "+str(round(prob,1))+"%")
        st.markdown("---")
        st.info("💡 " + FACTS[pred_class])

st.markdown("---")
st.markdown(
    "<p style=text-align:center;color:gray>"
    "Built by Harshal Mathur | "
    "CNN Model | 70.63% Accuracy | "
    "Beats research (53%)</p>",
    unsafe_allow_html=True
)
