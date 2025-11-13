# Streamlit app: Hybrid CNN (EfficientNetV2-M) + LLM (Replicate / IBM Granite)

import os
from io import BytesIO

import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import create_transform
from langchain_community.llms import Replicate

from disease_info import disease_info

# -------------------------------------------------------------------
# 0. Device
# -------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# 1. Model CNN (EfficientNetV2-M) + weight
# -------------------------------------------------------------------
@st.cache_resource
def load_model(model_path: str = "best_model_overall.pth"):
    """
    Muat arsitektur EfficientNetV2-M + head custom seperti di notebook
    dan load weight dari file .pth.
    """
    base_model = timm.create_model("efficientnetv2_rw_m", pretrained=False)

    # Head harus sama persis dengan training
    num_features = base_model.classifier.in_features
    num_classes = 11
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    )

    model = base_model.to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"File model '{model_path}' tidak ditemukan. "
            f"Pastikan sudah di-upload (mis. lewat Git LFS)."
        )

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_transform():
    """
    Transform inference yang sama dengan data_config di notebook.
    """
    data_config = {
        "input_size": (3, 320, 320),
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "crop_pct": 1.0,
        "crop_mode": "center",
    }
    return create_transform(**data_config, is_training=False)


# -------------------------------------------------------------------
# 2. LLM Replicate (IBM Granite 3.2 8B Instruct)
# -------------------------------------------------------------------
@st.cache_resource
def load_llm():
    """
    Inisialisasi LLM Replicate.
    Di Streamlit Cloud, set REPLICATE_API_TOKEN via menu 'Secrets'.
    """
    api_token = None

    # Prioritas secrets Streamlit (Cloud)
    try:
        api_token = st.secrets.get("REPLICATE_API_TOKEN", None)
    except Exception:
        api_token = None

    # Fallback ke environment variable lokal
    if not api_token:
        api_token = os.getenv("REPLICATE_API_TOKEN")

    if not api_token:
        raise ValueError(
            "REPLICATE_API_TOKEN belum di-set. "
            "Set di .streamlit/secrets.toml (lokal) atau di Secrets Streamlit Cloud."
        )

    model_name = "ibm-granite/granite-3.2-8b-instruct"

    llm = Replicate(
        model=model_name,
        replicate_api_token=api_token,
        model_kwargs={"max_new_tokens": 200, "temperature": 0.7},
    )
    return llm


# -------------------------------------------------------------------
# 3. Kelas target (harus sama dengan training)
# -------------------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

# Mapping untuk display yang lebih friendly
CLASS_DISPLAY = {
    "Apple___Apple_scab": {"plant": "🍎 Apple", "condition": "Apple Scab", "status": "diseased"},
    "Apple___Black_rot": {"plant": "🍎 Apple", "condition": "Black Rot", "status": "diseased"},
    "Apple___Cedar_apple_rust": {"plant": "🍎 Apple", "condition": "Cedar Apple Rust", "status": "diseased"},
    "Apple___healthy": {"plant": "🍎 Apple", "condition": "Healthy", "status": "healthy"},
    "Grape___Black_rot": {"plant": "🍇 Grape", "condition": "Black Rot", "status": "diseased"},
    "Grape___Esca_(Black_Measles)": {"plant": "🍇 Grape", "condition": "Esca (Black Measles)", "status": "diseased"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"plant": "🍇 Grape", "condition": "Leaf Blight", "status": "diseased"},
    "Grape___healthy": {"plant": "🍇 Grape", "condition": "Healthy", "status": "healthy"},
    "Potato___Early_blight": {"plant": "🥔 Potato", "condition": "Early Blight", "status": "diseased"},
    "Potato___Late_blight": {"plant": "🥔 Potato", "condition": "Late Blight", "status": "diseased"},
    "Potato___healthy": {"plant": "🥔 Potato", "condition": "Healthy", "status": "healthy"},
}


# -------------------------------------------------------------------
# 4. Utilitas image + prediksi
# -------------------------------------------------------------------
def load_image_from_input(uploaded_file, url: str):
    """
    Ambil gambar dari upload Streamlit atau URL.
    Return: PIL.Image, sumber_str
    """
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        return image, "uploaded file"
    elif url:
        resp = requests.get(url)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert("RGB")
        return image, url
    else:
        return None, None


def predict(image: Image.Image, model, transform):
    """
    CNN inference: image -> label + confidence.
    """
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

    label = CLASS_NAMES[idx.item()]
    confidence = float(conf.item())
    return label, confidence


# -------------------------------------------------------------------
# 5. Prompt builder + LLM explanation
# -------------------------------------------------------------------
def build_prompt(label: str, details: dict) -> str:
    general_symptoms = details.get("general_symptoms", [])
    distinguishing_features = details.get("distinguishing_features", [])
    early_actions = details.get("early_actions", [])

    prompt = f"""
You are an agronomy assistant that explains plant leaf diseases to farmers and agronomists.

Predicted disease label: {label}

General symptoms:
- """ + "\n- ".join(general_symptoms) + """

Distinguishing features:
- """ + "\n- ".join(distinguishing_features) + """

Early actions:
- """ + "\n- ".join(early_actions) + """

Write a concise explanation (2–3 short paragraphs) in English that:
1. Describes what this disease is and how it typically appears on leaves.
2. Highlights the key visual cues that match the symptoms above.
3. Suggests early, practical actions farmers can take (without naming specific commercial fungicide brands).

Avoid repeating the bullet lists verbatim; paraphrase them into natural text.
"""
    return prompt


def generate_explanation(llm, label: str) -> str:
    details = disease_info.get(label)
    if not details:
        return (
            f"No detailed disease information found in internal mapping "
            f"for label '{label}'."
        )

    prompt = build_prompt(label, details)

    try:
        # Replicate LLM dari LangChain mendukung .invoke()
        explanation = llm.invoke(prompt)
    except Exception as e:
        # Fallback kalau LLM error supaya app tetap jalan
        explanation = (
            f"LLM error: {e}\n\n"
            "Fallback summary based on internal rules:\n"
            f"- General symptoms: {', '.join(details.get('general_symptoms', []))}\n"
            f"- Distinguishing features: {', '.join(details.get('distinguishing_features', []))}\n"
            f"- Early actions: {', '.join(details.get('early_actions', []))}\n"
        )
    return explanation


# -------------------------------------------------------------------
# 6. Custom CSS
# -------------------------------------------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .stTitle {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
    }
    
    /* Card-like containers */
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Status badges */
    .status-healthy {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        border: 2px solid #c3e6cb;
    }
    
    .status-diseased {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        border: 2px solid #f5c6cb;
    }
    
    /* Confidence bar */
    .confidence-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        transition: width 0.5s ease;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Upload area */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# 7. Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Plant Disease AI Diagnosis",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()

    # Header
    st.markdown("<h1 style='text-align: center; color: #2c3e50; margin-bottom: 0;'>🌿 Plant Disease AI Diagnosis</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 1.2rem;'>Powered by CNN + Large Language Model</p>", unsafe_allow_html=True)
    
    st.markdown("---")

    # Sidebar: status model & LLM
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        
        try:
            model = load_model()
            transform = load_transform()
            st.success("✅ CNN Model Loaded")
            st.info(f"📱 Device: {device}")
        except Exception as e:
            st.error(f"❌ Error loading CNN model: {e}")
            st.stop()

        try:
            llm = load_llm()
            st.success("✅ LLM Ready (IBM Granite)")
        except Exception as e:
            st.warning(f"⚠️ LLM not available: {e}")
            llm = None
        
        st.markdown("---")
        st.markdown("### 📋 Supported Plants")
        st.markdown("- 🍎 Apple (4 classes)")
        st.markdown("- 🍇 Grape (4 classes)")
        st.markdown("- 🥔 Potato (3 classes)")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("This app uses **EfficientNetV2-M** for image classification and **IBM Granite** LLM for generating detailed disease explanations.")
        
        with st.expander("🔍 All Detectable Classes"):
            for idx, class_name in enumerate(CLASS_NAMES, 1):
                info = CLASS_DISPLAY[class_name]
                st.markdown(f"{idx}. {info['plant']} - {info['condition']}")

    # Main content
    col_upload, col_url = st.columns(2)
    
    with col_upload:
        st.markdown("### 📤 Upload Image")
        uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    with col_url:
        st.markdown("### 🔗 Or Use URL")
        url = st.text_input("Paste image URL", label_visibility="collapsed", placeholder="https://example.com/leaf.jpg")

    st.markdown("<br>", unsafe_allow_html=True)
    
    diagnose_col1, diagnose_col2, diagnose_col3 = st.columns([1, 2, 1])
    with diagnose_col2:
        diagnose_btn = st.button("🔬 Diagnose Plant", use_container_width=True)

    if diagnose_btn:
        image, source = load_image_from_input(uploaded, url)
        if image is None:
            st.error("⚠️ Please upload an image or provide a URL first.")
            return

        # Display image
        st.markdown("### 📸 Uploaded Image")
        img_col1, img_col2, img_col3 = st.columns([1, 2, 1])
        with img_col2:
            st.image(image, caption=f"Source: {source}", use_container_width=True)

        # Prediction
        with st.spinner("🔄 Analyzing image with AI..."):
            model = load_model()
            transform = load_transform()
            label, confidence = predict(image, model, transform)

        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Diagnosis Results")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            info = CLASS_DISPLAY[label]
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"#### {info['plant']}")
            st.markdown(f"### {info['condition']}")
            
            status_class = "status-healthy" if info['status'] == 'healthy' else "status-diseased"
            status_text = "✅ Healthy" if info['status'] == 'healthy' else "⚠️ Disease Detected"
            st.markdown(f'<div class="{status_class}">{status_text}</div>', unsafe_allow_html=True)
            
            st.markdown("#### Confidence Level")
            confidence_pct = confidence * 100
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_pct}%">
                    {confidence_pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Technical Label:** `{label}`")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Quick Stats")
            
            if confidence >= 0.9:
                conf_status = "🟢 Very High"
            elif confidence >= 0.7:
                conf_status = "🟡 High"
            elif confidence >= 0.5:
                conf_status = "🟠 Moderate"
            else:
                conf_status = "🔴 Low"
            
            st.metric("Confidence Level", f"{confidence:.2%}", conf_status)
            st.metric("Plant Type", info['plant'])
            st.metric("Health Status", info['condition'])
            
            st.markdown('</div>', unsafe_allow_html=True)

        # LLM Explanation
        if llm is not None:
            st.markdown("---")
            st.markdown("### 🤖 AI-Generated Explanation")
            
            with st.spinner("💭 Generating detailed explanation..."):
                explanation = generate_explanation(llm, label)
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(explanation)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show disease details if available
            details = disease_info.get(label)
            if details:
                with st.expander("📋 View Detailed Information"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown("**General Symptoms:**")
                        for symptom in details.get("general_symptoms", []):
                            st.markdown(f"- {symptom}")
                    
                    with col_b:
                        st.markdown("**Distinguishing Features:**")
                        for feature in details.get("distinguishing_features", []):
                            st.markdown(f"- {feature}")
                    
                    with col_c:
                        st.markdown("**Early Actions:**")
                        for action in details.get("early_actions", []):
                            st.markdown(f"- {action}")
        else:
            st.markdown("---")
            st.info("💡 **LLM not configured.** Set REPLICATE_API_TOKEN to enable detailed AI explanations.")

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Made with ❤️ using Streamlit | EfficientNetV2-M + IBM Granite LLM</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
