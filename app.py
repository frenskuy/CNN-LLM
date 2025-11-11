# app.py
# =============================================================
# Minimal Streamlit UI: upload image + "Run Classification"
# CNN: EfficientNetV2-M (timm) – expects best_model_overall.pth
# LLM explanation is optional & safe-fallback if transformers/pipeline unavailable
# Language: English
# =============================================================

import os
from io import BytesIO

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import timm
from timm.data import create_transform

# ----------------------------
# Helpers
# ----------------------------
def clean_label(label: str) -> str:
    """Human-friendly: 'Apple___Black_rot' -> 'Apple → Black rot'."""
    return label.replace("___", " → ").replace("_", " ")

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Disease knowledge mapping (trim/extend as needed)
# ----------------------------
disease_info = {
    "Grape___Black_rot": {
        "general_symptoms": [
            "Large brown spots with target-like concentric rings on leaves",
            "Berries turn black and shrivel with firm brown lesions",
            "Canes may show elongated brown lesions"
        ],
        "distinguishing_features": [
            "Concentric ring (target) pattern on leaves",
            "Mummified, hard black berries"
        ],
        "early_actions": [
            "Sanitation: remove infected tissues",
            "Apply preventive fungicide if needed"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "general_symptoms": [
            "Interveinal chlorosis and necrosis (sometimes tiger-stripe)",
            "Berries may crack and show dark spots",
            "Trunk/wood dark streaking"
        ],
        "distinguishing_features": [
            "Tiger-stripe chlorosis on leaves",
            "Trunk disease involvement"
        ],
        "early_actions": [
            "Prune and remove infected wood",
            "Consider trunk renewal strategies"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "general_symptoms": [
            "Large irregular brown necrotic spots on leaves",
            "Spots may coalesce into blighted areas"
        ],
        "distinguishing_features": [
            "Large irregular spots without target rings"
        ],
        "early_actions": [
            "Remove heavily infected leaves",
            "Improve airflow; fungicide if severe"
        ]
    },
    "Grape___healthy": {
        "general_symptoms": ["Uniform green leaves"],
        "distinguishing_features": ["No visible disease symptoms"],
        "early_actions": ["Maintain good cultural practices"]
    },
    "Apple___Apple_scab": {
        "general_symptoms": [
            "Olive-green to brown velvety lesions on leaves/fruit",
            "Leaf curling or deformation"
        ],
        "distinguishing_features": [
            "Velvety scab-like lesions, olive-green on young leaves"
        ],
        "early_actions": [
            "Remove fallen leaves; preventive fungicide if needed"
        ]
    },
    "Apple___Black_rot": {
        "general_symptoms": [
            "Leaf spots with purple margins and tan centers",
            "Pycnidia may appear in lesions"
        ],
        "distinguishing_features": [
            "Frog-eye leaf spot",
            "Fruit rot with concentric rings"
        ],
        "early_actions": [
            "Prune infected twigs and mummified fruit"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "general_symptoms": [
            "Yellow-orange spots on upper leaf surfaces",
            "Tube-like aecia on the underside"
        ],
        "distinguishing_features": [
            "Orange rust spots with spore structures",
            "Juniper is an alternate host"
        ],
        "early_actions": [
            "Remove nearby junipers if feasible",
            "Apply fungicide early season if risk"
        ]
    },
    "Apple___healthy": {
        "general_symptoms": ["Uniform green leaves"],
        "distinguishing_features": ["No disease symptoms"],
        "early_actions": ["Maintain sanitation and monitoring"]
    },
    "Potato___Early_blight": {
        "general_symptoms": [
            "Brown spots with target-like concentric rings",
            "Yellow halos around lesions"
        ],
        "distinguishing_features": [
            "Target-like rings on older leaves"
        ],
        "early_actions": [
            "Remove infected debris; rotate crops"
        ]
    },
    "Potato___Late_blight": {
        "general_symptoms": [
            "Water-soaked lesions; rapid blight in cool/wet",
            "Whitish growth under leaf in humid weather"
        ],
        "distinguishing_features": [
            "Very rapid spread; water-soaked lesions"
        ],
        "early_actions": [
            "Immediate sanitation + protective fungicide",
            "Avoid overhead irrigation"
        ]
    },
    "Potato___healthy": {
        "general_symptoms": ["Green leaves without necrosis/chlorosis"],
        "distinguishing_features": ["No disease symptoms"],
        "early_actions": ["Maintain crop health and monitoring"]
    }
}

# ----------------------------
# Transform & class names (must match training)
# ----------------------------
data_config = {
    'input_size': (3, 320, 320),
    'interpolation': 'bicubic',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'crop_pct': 1.0,
    'crop_mode': 'center'
}
transform_val = create_transform(**data_config, is_training=False)

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
]

# ----------------------------
# Model loader (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path: str = "best_model_overall.pth"):
    """Load EfficientNetV2-M + custom classifier head; return eval-mode model and device."""
    dev = device
    base_model = timm.create_model('efficientnetv2_rw_m', pretrained=False)
    num_features = base_model.classifier.in_features
    num_classes = 11
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    model = base_model.to(dev)
    state = torch.load(model_path, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model, dev

# ----------------------------
# Optional LLM loader (safe if transformers missing)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_llm(model_name: str = "google/flan-t5-base"):
    """
    Try to import transformers.pipeline only when needed.
    If unavailable (e.g., Python 3.13 incompatibility), return None and use fallback text.
    """
    try:
        from transformers import pipeline as hf_pipeline  # may fail in some envs
    except Exception as e:
        st.warning(f"Transformers not available ({e}); using knowledge-based explanation.")
        return None

    try:
        pipe = hf_pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_length=200
        )
        return pipe
    except Exception as e:
        st.warning(f"LLM load failed ({e}); using knowledge-based explanation.")
        return None

# ----------------------------
# Inference utilities
# ----------------------------
def predict_pil(image_pil: Image.Image, model, transform, class_names, device):
    """Predict from a PIL image and return (label, confidence)."""
    try:
        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            return class_names[idx], float(probs[idx].item())
    except Exception:
        return None, None

def generate_explanation(predicted_label: str, disease_info_map: dict, llm_pipe=None) -> str:
    """LLM explanation with robust fallback."""
    info = disease_info_map.get(predicted_label, {})
    general = info.get("general_symptoms", [])
    distinct = info.get("distinguishing_features", [])
    actions = info.get("early_actions", [])

    pretty_label = clean_label(predicted_label)
    fallback_text = (
        f"Predicted class: {pretty_label}. "
        f"General symptoms: {('; '.join(general)) or 'n/a'}. "
        f"Distinguishing features: {('; '.join(distinct)) or 'n/a'}. "
        f"Early actions: {('; '.join(actions)) or 'n/a'}."
    )

    if llm_pipe is None:
        return fallback_text

    try:
        prompt = (
            "You are an agronomy assistant. Explain briefly and factually why this leaf image "
            f"is predicted as '{pretty_label}'. Present three parts: "
            "(1) general symptoms, (2) distinguishing features, (3) recommended early actions. "
            "Use clear English, 3–6 sentences. Do not use underscores in class names.\n\n"
            f"General symptoms (hints): {'; '.join(general)}\n"
            f"Distinguishing features (hints): {'; '.join(distinct)}\n"
            f"Early actions (hints): {'; '.join(actions)}"
        )
        res = llm_pipe(prompt, num_return_sequences=1, num_beams=4, do_sample=False)
        return res[0]["generated_text"]
    except Exception as e:
        return f"LLM explanation unavailable ({e}). " + fallback_text

# ----------------------------
# Streamlit UI (minimal)
# ----------------------------
st.set_page_config(page_title="Leaf Disease Classification", page_icon="🌿", layout="centered")

st.title("🌿 Leaf Disease Classification")
st.caption("Upload a leaf image, then run classification. LLM explanation is optional and falls back to curated knowledge if unavailable.")

image_file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
use_llm = st.checkbox("Enable LLM explanation (optional)", value=False)
run_btn = st.button("🚀 Run Classification")

if run_btn:
    if image_file is None:
        st.error("Please upload an image first.")
        st.stop()

    # Load model
    try:
        with st.spinner("Loading model..."):
            model, dev = load_cnn_model("best_model_overall.pth")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    # Read image
    with st.spinner("Reading image..."):
        try:
            image = Image.open(image_file).convert("RGB")
        except Exception as e:
            st.error(f"Failed to read image: {e}")
            st.stop()

    # Predict
    with st.spinner("Running inference..."):
        pred, conf = predict_pil(image, model, transform_val, class_names, dev)
        if pred is None:
            st.error("Prediction failed.")
            st.stop()

    # LLM (optional) – returns None if not available
    with st.spinner("Preparing explanation..."):
        llm_pipe = load_llm() if use_llm else None
        explanation = generate_explanation(pred, disease_info, llm_pipe=llm_pipe)

    # Display results
    pretty_label = clean_label(pred)
    st.subheader("🎯 Prediction")
    st.metric("Top-1 Prediction", pretty_label, f"{conf*100:.2f}%")

    st.subheader("🧠 Explanation")
    # Use plain text to avoid Markdown underscore issues
    st.text(explanation)

    st.subheader("🖼️ Input Image")
    st.image(image, caption=f"Prediction: {pretty_label} ({conf*100:.2f}%)", width="stretch")
