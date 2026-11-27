import cv2
import numpy as np
import streamlit as st
from predict import predict_sign

# ---------- Page settings ----------
st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")

# ---------- Simple styling ----------
st.markdown(
    """
    <style>
    .title-box {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 24px;
        border-radius: 14px;
        text-align: center;
        margin-bottom: 20px;
    }
    .title-box h1 { color: white; margin: 0; font-size: 2rem; }
    .title-box p  { color: #dbe6ff; margin: 6px 0 0 0; }
    .result-card {
        background: rgba(42, 82, 152, 0.12);
        border-left: 6px solid #2a5298;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .result-card h3 { margin: 0 0 6px 0; }
    </style>
    <div class="title-box">
        <h1>Traffic Sign Recognition System</h1>
        <p>Upload a traffic sign image and the CNN model will identify it</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Used to reset the uploader ----------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploaded = st.file_uploader(
    "Choose a traffic sign image (JPG, JPEG or PNG)",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}",
)

if uploaded is None:
    st.info("Please upload an image to get started.")
else:
    # Read the uploaded file with OpenCV
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read this image. Please try another file.")
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Predict
        with st.spinner("Analysing the image..."):
            class_id, name, meaning, confidence = predict_sign(img_bgr)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded image")
            st.image(img_rgb, width=260)

        with col2:
            st.subheader("Prediction")
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>{name}</h3>
                    <p>{meaning}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write(f"**Confidence:** {confidence:.2f} %")
            st.progress(min(int(confidence), 100))

        if confidence < 70:
            st.warning(
                "The model is not very sure about this image. "
                "Try a clearer, closer picture of a single sign."
            )

        with st.expander("See the preprocessed image (what the model sees)"):
            small = cv2.resize(img_rgb, (32, 32))
            st.image(small, width=128, caption="Resized to 32 x 32 pixels")

        if st.button("Upload another image"):
            st.session_state.uploader_key += 1
            st.rerun()

# ---------- About section ----------
st.divider()

with st.expander("About this project: Computer Vision and Intelligent Transportation"):
    st.markdown(
        """
        **Computer Vision**

        Computer vision is the field that teaches computers to understand images.
        In this project:
        1. **OpenCV** reads the image, converts its colours and resizes it to 32 x 32 pixels.
        2. A **CNN (Convolutional Neural Network)** looks at the pixels and finds patterns
           such as edges, shapes and colours.
        3. The model gives a score for each of the 43 sign types, and the highest score
           is shown as the prediction with its confidence.

        **Intelligent Transportation Systems (ITS)**

        ITS uses technology to make roads safer and traffic smarter. Recognising traffic
        signs automatically is one part of this. It is useful in driver assistance
        systems, which can warn a driver about a speed limit or a stop sign, and in
        self-driving vehicles, which must read signs to follow road rules.

        **About this model**

        - Dataset: GTSRB (German Traffic Sign Recognition Benchmark), 43 sign types
        - Accuracy on the official Test set (12,630 images): 95.76 %
        - Limitation: it can only recognise these 43 signs. Other objects, and very
          dark or blurry images, can give wrong results.
        """
    )

st.caption("Mini Project - Traffic Sign Recognition using CNN (GTSRB dataset)")