import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Page config
st.set_page_config(
    page_title="Handwriting Origin Identifier",
    page_icon="✍️",
    layout="centered"
)

# Title
st.title("✍️ Handwriting Origin Identifier")
st.markdown("### Upload a handwriting image to identify its origin!")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    model_path = "final_model.keras"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_model()

COUNTRIES = ["🇮🇳 Indian", "🇺🇸 American", "🇨🇳 Chinese", "🇯🇵 Japanese"]
COLORS    = ["#FF9933", "#3C3B6E", "#DE2910", "#BC002D"]

if model is None:
    st.error("Model not found! Please check the model file.")
else:
    st.success("Model loaded successfully!")

    # File uploader
    st.markdown("### Upload Handwriting Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Show uploaded image
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Your Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        # Preprocess image
        img = image.convert("L")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 1)

        # Predict
        with st.spinner("Analyzing handwriting..."):
            predictions = model.predict(img_array, verbose=0)
            pred_class  = np.argmax(predictions[0])
            confidence  = predictions[0][pred_class] * 100

        with col2:
            st.markdown("#### Result")
            st.markdown(
                f"<h2 style=color:{COLORS[pred_class]}>"
                f"{COUNTRIES[pred_class]}</h2>",
                unsafe_allow_html=True
            )
            st.metric("Confidence", f"{confidence:.1f}%")

        # Show all probabilities
        st.markdown("---")
        st.markdown("#### Confidence For Each Country")

        for i, (country, color) in enumerate(
                zip(COUNTRIES, COLORS)):
            prob = predictions[0][i] * 100
            st.markdown(f"**{country}**")
            st.progress(int(prob))
            st.caption(f"{prob:.1f}%")

        # Fun fact
        st.markdown("---")
        st.markdown("#### Did You Know?")
        facts = {
            0: "Indian handwriting tends to be upright with round, uniform letters influenced by local scripts!",
            1: "American handwriting often shows a right-leaning slant from cursive writing traditions!",
            2: "Chinese writers bring brush-stroke precision to their English handwriting!",
            3: "Japanese handwriting is extremely consistent in size due to disciplined school training!"
        }
        st.info(facts[pred_class])

# Footer
st.markdown("---")
st.markdown(
    "Built with TensorFlow & Streamlit | "
    "Handwriting Origin Classifier Project"
)
'''

# Save the app file
app_path = base_path + '/app.py'
with open(app_path, 'w') as f:
    f.write(app_code)

print("Web app file created!")
print("Location: " + app_path)
