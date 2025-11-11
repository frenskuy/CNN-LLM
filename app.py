# app.py
# =============================================================
# Minimal Streamlit UI: upload image + "Run Classification" button
# Model must be in working dir as: best_model_overall.pth
# Language: English
# =============================================================

import os
from io import BytesIO

import streamlit as st
import requests

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
    """Make class names human-friendly: 'Apple___Black_rot' -> 'Apple → Black rot'."""
    return label.replace("___", " → ").replace("_", " ")

def escape_markdown(s: str) -> str:
    """Escape common Markdown characters if you ever use st.markdown."""
    for ch in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!"):
        s = s.replace(ch, "\\" + ch)
    return s

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Disease knowledge mapping
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
            "Interveinal chlorosis and necrosis on leaves, sometimes tiger-stripe patterns",
            "Berries may crack and show dark spots",
            "Wood shows dark streaking (trunk disease)"
        ],
        "distinguishing_features": [
            "Tiger-stripe (striped chlorosis) pattern on leaves",
            "Trunk/wood disease involvement"
        ],
        "early_actions": [
            "Prune and remove infected wood",
            "Improve vine vigor, consider trunk renewal strategies"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "general_symptoms": [
            "Large brown necrotic spots with irregular margins on leaves",
            "Spots can coalesce into extensive blighted areas",
            "Damage reduces photosynthesis"
        ],
        "distinguishing_features": [
            "Large irregular spots without target rings",
            "Severe blight patterns on older leaves"
        ],
        "early_actions": [
            "Remove heavily infected leaves",
            "Improve airflow, consider fungicide if severe"
        ]
    },
    "Grape___healthy": {
        "general_symptoms": [
            "Uniform green leaves, no lesions or chlorosis"
        ],
        "distinguishing_features": [
            "No visible disease symptoms or necrotic areas"
        ],
        "early_actions": [
            "Maintain regular monitoring and good cultural practices"
        ]
    },
    "Apple___Apple_scab": {
        "general_symptoms": [
            "Olive-green to brown velvety lesions on leaves and fruit",
            "Leaf curling or deformation"
        ],
        "distinguishing_features": [
            "Velvety scab-like lesions, olive-green on young leaves"
        ],
        "early_actions": [
            "Remove fallen leaves, apply fungicide preventively if necessary"
        ]
    },
    "Apple___Black_rot": {
        "general_symptoms": [
            "Leaf spots with purple margins and tan centers",
            "Fruiting bodies (pycnidia) may appear in lesions"
        ],
        "distinguishing_features": [
            "Frog-eye leaf spot pattern",
            "Fruit rot with concentric rings on apples"
        ],
        "early_actions": [
            "Prune infected twigs and mummified fruit",
            "Consider fungicide during critical periods"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "general_symptoms": [
            "Yellow-orange spots on upper leaf surfaces",
            "Tube-like aecia on underside of leaves"
        ],
        "distinguishing_features": [
            "Orange rust spots with spore structures",
            "Requires juniper as alternate host"
        ],
        "early_actions": [
            "Remove nearby junipers if feasible",
            "Fungicide applications in early season may help"
        ]
    },
    "Apple___healthy": {
        "general_symptoms": [
            "Uniform green leaves without significant lesions"
        ],
        "distinguishing_features": [
            "No visible disease symptoms"
        ],
        "early_actions": [
            "Maintain orchard sanitation and monitoring"
        ]
    },
    "Potato___Early_blight": {
        "general_symptoms": [
            "Brown spots with concentric rings (target-like) on leaves",
            "Yellow halos around lesions"
        ],
        "distinguishing_features": [
            "Target-like concentric rings on older leaves"
        ],
        "early_actions": [
            "Remove infected debris, rotate crops",
            "Apply fungicide if pressure is high"
        ]
    },
    "Potato___Late_blight": {
        "general_symptoms": [
            "Water-soaked lesions, rapid blight under cool/wet conditions",
            "Whitish fungal growth under leaf in humid weather"
        ],
        "distinguishing_features": [
            "Rapidly spreading blight, water-soaked lesions"
        ],
        "early_actions": [
            "Immediate sanitation and protective fungicide",
            "Avoid overhead irrigation, monitor closely"
        ]
    },
    "Potato___healthy": {
        "general_symptoms": [
            "Green leaves without necrotic lesions or chlorosis"
        ],
        "distinguishing_features": [
            "No disease symptoms"
        ],
        "early_actions": [
            "Maintain general crop health and monitoring"
        ]
    }
}

# ----------------------------
# Transform & class names (match training)
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
# Model & LLM loaders
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

@st.cache_resource(show_spinner=False)
def load_llm(model_name: str = "google/flan-t5-base"):
    """Load a text2text LLM; if it fails (no internet/weights), calls will fallback."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_length=200
        )
        return pipe
    except Exception:
        return None  # We'll fallback to knowledge-based text

# ----------------------------
# Inference utilities
# ----------------------------
def test_custom_image_pil(image_pil: Image.Image, model, transform, class_names, device):
    """Predict from an in-memory PIL image."""
    try:
        input_tensor = transform(image_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted_class_index = torch.max(probabilities, 1)
        predicted_class_name = class_names[predicted_class_index.item()]
        confidence = probabilities[0, predicted_class_index.item()].item()
        return predicted_class_name, confidence
    except Exception as e:
        return None, None

def generate_explanation(predicted_label: str, disease_info_map: dict, llm_pipe=None) -> str:
    """LLM explanation with safe fallback."""
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
st.title("🌿 Leaf Disease Classification (CNN + LLM Explanation)")

st.markdown("Upload a **leaf image** and click **Run Classification**. The app will load `best_model_overall.pth` automatically.")

image_file = st.file_uploader("Upload image (JPG/PNG)", type=["jpg", "jpeg", "png"])
run_btn = st.button("🚀 Run Classification")

if run_btn:
    if image_file is None:
        st.error("Please upload an image first.")
    else:
        # Load model & LLM (cached)
        try:
            with st.spinner("Loading model..."):
                model, dev = load_cnn_model("best_model_overall.pth")
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

        with st.spinner("Loading image..."):
            try:
                image = Image.open(image_file).convert("RGB")
            except Exception as e:
                st.error(f"Failed to read image: {e}")
                st.stop()

        with st.spinner("Running inference..."):
            pred, conf = test_custom_image_pil(image, model, transform_val, class_names, dev)
            if pred is None:
                st.error("Prediction failed.")
                st.stop()

        # Try LLM (cached); fallback text if unavailable
        with st.spinner("Generating explanation..."):
            llm_pipe = load_llm()  # may be None if loading fails
            explanation = generate_explanation(pred, disease_info, llm_pipe=llm_pipe)

        # Display results
        pretty_label = clean_label(pred)
        st.subheader("🎯 Prediction")
        st.metric("Top-1 Prediction", pretty_label, f"{conf*100:.2f}%")

        st.subheader("🧠 Explanation")
        st.text(explanation)  # safer than markdown for underscores

        st.subheader("🖼️ Input Image")
        st.image(image, caption=f"Prediction: {pretty_label} ({conf*100:.2f}%)", use_container_width=True)
