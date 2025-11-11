import torch
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Import util
from classificationllm import (
    load_model as core_load_model,
    load_llm as core_load_llm,
    get_transform as core_get_transform,
    predict_image as core_predict_image,
    generate_explanation as core_generate_explanation,
    CLASS_NAMES,
    DISEASE_INFO,
)

# ---------------------------------------
# Page configuration (harus paling atas)
# ---------------------------------------
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# Samakan nama variabel dengan UI kamu
class_names = CLASS_NAMES
disease_info = DISEASE_INFO

# -------------------------------------------------------------
# Cache resource untuk komponen berat
# -------------------------------------------------------------
@st.cache_resource
def load_model():
    """Wrapper cache yang memanggil loader di tftf.py."""
    return core_load_model("best_model_overall.pth")  # (model, device)

@st.cache_resource
def load_llm():
    """Wrapper cache LLM pipeline."""
    return core_load_llm()

@st.cache_resource
def get_transform():
    """Wrapper cache transform validasi."""
    return core_get_transform()

# Pembungkus agar tanda tangan fungsi sama
def predict_image(image, model, transform, device):
    """Panggil inferensi dari tftf.py, dengan penanganan error UI."""
    try:
        predicted_label, confidence, probs = core_predict_image(image, model, transform, device)
        return predicted_label, confidence, probs
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def generate_explanation(predicted_label, confidence, llm_pipeline):
    """Panggil penjelasan LLM dari tftf.py (fallback jika gagal)."""
    try:
        return core_generate_explanation(predicted_label, confidence, llm_pipeline)
    except Exception as e:
        st.warning(f"Could not generate LLM explanation: {e}")
        det = disease_info.get(predicted_label, {})
        general = det.get("general_symptoms", [])
        distin = det.get("distinguishing_features", [])
        return (
            f"The model predicted '{predicted_label}' with {confidence:.1%} confidence. "
            f"General symptoms: {'; '.join(general)}. "
            f"Distinguishing features: {'; '.join(distin)}."
        )

# -------------------------------------------------------------
# Main App
# -------------------------------------------------------------
def main():
    st.title("🌿 Plant Disease Classification System")
    st.write("Upload an image of a plant leaf to detect diseases in Apple, Grape, and Potato plants.")
    
    # Load resource utama (cached)
    with st.spinner("Loading model..."):
        model, device = load_model()
        llm_pipe = load_llm()
        transform = get_transform()
    st.success("Model loaded successfully!")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses a deep learning model (EfficientNetV2-M) "
        "to classify plant diseases in Apple, Grape, and Potato leaves. "
        "It provides detailed explanations using an AI language model."
    )
    
    st.sidebar.header("Supported Classes")
    for cls in class_names:
        st.sidebar.text(f"• {cls.replace('___', ' - ')}")
    
    # Pilihan input
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
    
    if image is not None:
        # Tampilkan gambar
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
        
        # Tombol analisis
        if st.button("🔍 Analyze Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_label, confidence, all_probs = predict_image(image, model, transform, device)
            
            if predicted_label is not None:
                with col2:
                    st.subheader("Prediction Results")
                    st.metric("Predicted Class", predicted_label.replace('___', ' - '))
                    st.metric("Confidence", f"{confidence:.2%}")
                    st.progress(confidence)  # nilai 0..1
                
                # Top 3 prediksi
                st.subheader("Top 3 Predictions")
                top3_probs, top3_indices = torch.topk(all_probs, 3)
                cols = st.columns(3)
                for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
                    with cols[i]:
                        st.metric(
                            f"#{i+1}: {class_names[int(idx)].replace('___', ' - ')}",
                            f"{prob.item():.2%}"
                        )
                
                # Penjelasan LLM
                st.subheader("📋 Detailed Analysis")
                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(predicted_label, confidence, llm_pipe)
                st.write(explanation)
                
                # Panel info penyakit
                if predicted_label in disease_info:
                    st.subheader("🔬 Disease Information")
                    disease_details = disease_info[predicted_label]
                    
                    with st.expander("General Symptoms"):
                        for symptom in disease_details.get("general_symptoms", []):
                            st.write(f"• {symptom}")
                    
                    with st.expander("Distinguishing Features"):
                        for feature in disease_details.get("distinguishing_features", []):
                            st.write(f"• {feature}")
                    
                    with st.expander("Early Actions"):
                        for action in disease_details.get("early_actions", []):
                            st.write(f"• {action}")

if __name__ == "__main__":
    main()

