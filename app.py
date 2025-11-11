# app.py
import torch
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

from tftf import (
    load_model, load_llm, get_transform,
    predict_image, generate_explanation,
    CLASS_NAMES, DISEASE_INFO
)

# --------------------
# Page config
# --------------------
st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")
st.title("🌿 Plant Disease Classification System")
st.write("Upload an image of a plant leaf to detect diseases in Apple, Grape, and Potato plants.")

# --------------------
# Cache wrappers
# --------------------
@st.cache_resource
def cached_model(weights_path: str = "best_model_overall.pth"):
    # load_model mengembalikan (model, device)
    return load_model(weights_path)

@st.cache_resource
def cached_llm():
    return load_llm()

@st.cache_resource
def cached_transform():
    return get_transform()

# Init once
with st.spinner("Loading model..."):
    model, device = cached_model("best_model_overall.pth")
    llm_pipe = cached_llm()
    transform = cached_transform()
st.success("Model loaded successfully!")

# --------------------
# Sidebar
# --------------------
st.sidebar.header("About")
st.sidebar.info(
    "This application uses a deep learning model (EfficientNetV2-M) "
    "to classify plant diseases in Apple, Grape, and Potato leaves. "
    "It provides detailed explanations using an AI language model."
)

st.sidebar.header("Supported Classes")
for cls in CLASS_NAMES:
    st.sidebar.text(f"• {cls.replace('___', ' - ')}")

# --------------------
# Image input
# --------------------
upload_option = st.radio("Choose input method:", ["Upload Image", "Image URL"])
image = None

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    image_url = st.text_input("Enter image URL:")
    if image_url:
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")

# --------------------
# Inference UI
# --------------------
if image is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    if st.button("🔍 Analyze Disease", type="primary"):
        with st.spinner("Analyzing image..."):
            predicted_label, confidence, probs = predict_image(image, model, transform, device)

        with col2:
            st.subheader("Prediction Results")
            st.metric("Predicted Class", predicted_label.replace('___', ' - '))
            st.metric("Confidence", f"{confidence:.2%}")
            st.progress(confidence)

        # Top-3
        st.subheader("Top 3 Predictions")
        top3_probs, top3_indices = torch.topk(probs, 3)
        cols = st.columns(3)
        for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
            with cols[i]:
                st.metric(f"#{i+1}: {CLASS_NAMES[int(idx)].replace('___', ' - ')}", f"{prob.item():.2%}")

        # LLM explanation
        st.subheader("📋 Detailed Analysis")
        try:
            explanation = generate_explanation(predicted_label, confidence, llm_pipe)
        except Exception as e:
            st.warning(f"Could not generate LLM explanation: {e}")
            det = DISEASE_INFO.get(predicted_label, {})
            st.write(
                f"The model predicted '{predicted_label}' with {confidence:.1%} confidence. "
                f"General symptoms: {'; '.join(det.get('general_symptoms', []))}. "
                f"Distinguishing features: {'; '.join(det.get('distinguishing_features', []))}."
            )
        else:
            st.write(explanation)

        # Knowledge panel
        if predicted_label in DISEASE_INFO:
            st.subheader("🔬 Disease Information")
            det = DISEASE_INFO[predicted_label]
            with st.expander("General Symptoms"):
                for s in det.get("general_symptoms", []):
                    st.write(f"• {s}")
            with st.expander("Distinguishing Features"):
                for f in det.get("distinguishing_features", []):
                    st.write(f"• {f}")
            with st.expander("Early Actions"):
                for a in det.get("early_actions", []):
                    st.write(f"• {a}")
