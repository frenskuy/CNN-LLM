# app.py
# =============================================================
# Streamlit App derived from notebook structure: tftf.ipynb
# The code follows the same sections/cell order as in the .ipynb,
# with minimal adaptations for Streamlit (caching + UI).
# Language: English.
# =============================================================

# (Cell 0) Imports, device selection
import os
from io import BytesIO
import requests
import streamlit as st

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import timm  # Import timm for EfficientNetV2-M
from timm.data import create_transform  # Import create_transform
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import shutil
import torch.nn.functional as F  # Import F for softmax

# Select CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# (Cell 1) Disease knowledge mapping
# --- Disease Knowledge Mapping ---
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

# (Cell 6) Validation transform and class names (defined in notebook)
# --- Recreate the validation transform and class names ---
# Adjust these values to match the data_config used during training
data_config = {
    'input_size': (3, 320, 320),
    'interpolation': 'bicubic',
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
    'crop_pct': 1.0,
    'crop_mode': 'center'
}
transform_val = create_transform(**data_config, is_training=False)

# Must match training order
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
]

# (Cell 2) Load the saved model (wrapped in a Streamlit cache)
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str, device_str: str | None = None):
    """Load the exact training architecture and weights; return model + device."""
    dev = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
    device_local = torch.device(dev)

    # --- Load the Saved Model (wrapped for Streamlit caching) ---
    # Make sure 'best_model_overall.pth' is available in your environment
    # model_path = 'best_model_overall.pth'  # Or the path to your saved model

    # Re-create the model architecture
    # Make sure the architecture matches the one used during training (EfficientNetV2-M + custom head)
    base_model = timm.create_model('efficientnetv2_rw_m', pretrained=False)  # No pretrained weights needed here

    # Get the number of input features for the classifier - MUST match training
    num_features = base_model.classifier.in_features

    # Replace the classifier head - MUST match training
    num_classes = 11  # Make sure this matches the number of classes in your training dataset
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )

    # Load the saved state dictionary
    model = base_model.to(device_local)
    state_dict = torch.load(model_path, map_location=device_local)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode

    return model, device_local

# (Cell 3) Initialize LLM (wrapped in a Streamlit cache)
@st.cache_resource(show_spinner=False)
def load_llm_cached(model_name: str = "google/flan-t5-base"):
    # --- Initialize LLM (wrapped for Streamlit caching) ---
    from transformers import pipeline
    llm_pipe = pipeline(
        "text2text-generation",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
        max_length=200
    )
    return llm_pipe

# (Cell 4) Utility: test_custom_image (kept the same function signature)
# --- Utility: test_custom_image (unchanged) ---
def test_custom_image(image_path_or_url, model, transform, class_names, device):
    """
    Loads an image from a local path or URL, applies transformations,
    and performs prediction using the provided model.
    """
    try:
        if image_path_or_url.startswith('http') or image_path_or_url.startswith('https'):
            response = requests.get(image_path_or_url)
            response.raise_for_status()  # Raise an exception for bad status codes
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            if not os.path.exists(image_path_or_url):
                print(f"Error: Image file not found at {image_path_or_url}")
                return None, None, None
            image = Image.open(image_path_or_url).convert('RGB')

        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted_class_index = torch.max(probabilities, 1)

        predicted_class_name = class_names[predicted_class_index.item()]
        confidence = probabilities[0, predicted_class_index.item()].item()
        return predicted_class_name, confidence, image  # Return the original image
    except Exception as e:
        print(f"An error occurred during image loading or prediction: {e}")
        return None, None, None

# (Cell 5) Utility: predict_and_explain (Streamlit wrapper that returns values)
def predict_and_explain_streamlit(image_path_or_url, model, transform, class_names, device, disease_info_map, llm_pipeline):
    """
    Streamlit-friendly wrapper:
    - Loads and predicts using test_custom_image
    - Builds an LLM explanation using the same logic as the notebook
    - Returns: (predicted_label, confidence, original_image, explanation_text)
    """
    predicted_label, confidence, original_image = test_custom_image(
        image_path_or_url, model, transform, class_names, device
    )

    if predicted_label is None or confidence is None or original_image is None:
        return None, None, None, "Prediction failed; cannot generate explanation."

    info = disease_info_map.get(predicted_label, {})
    general_symptoms = info.get("general_symptoms", [])
    distinguishing_features = info.get("distinguishing_features", [])
    early_actions = info.get("early_actions", [])

    fallback_text = (
        f"Predicted class: {predicted_label}. "
        f"General symptoms: {'; '.join(general_symptoms)}. "
        f"Distinguishing features: {'; '.join(distinguishing_features)}. "
        f"Early actions: {'; '.join(early_actions)}."
    )

    explanation = None
    if llm_pipeline is not None:
        try:
            prompt = (
                "You are an agronomy assistant. Explain briefly and factually why this leaf image "
                f"is predicted as '{clean_label(predicted_label)}'. Present three parts: "
                "(1) general symptoms, (2) distinguishing features, (3) recommended early actions. "
                "Use clear English, 3–6 sentences. "
                "Do not include underscores; write class names as plain words (replace '___' with ' → ' and '_' with spaces)."
                "\n\n"
                f"General symptoms (hints): {'; '.join(general_symptoms)}\n"
                f"Distinguishing features (hints): {'; '.join(distinguishing_features)}\n"
                f"Early actions (hints): {'; '.join(early_actions)}"
            )
            res = llm_pipeline(prompt, num_return_sequences=1)
            explanation = res[0]["generated_text"]
        except Exception as e:
            explanation = f"LLM explanation unavailable ({e}). " + fallback_text
    else:
        explanation = fallback_text

    return predicted_label, confidence, original_image, explanation

# =======================
# Streamlit UI (replaces Cell 7 manual test)
# =======================
st.set_page_config(page_title="CNN + LLM Explainability", page_icon="🌿", layout="wide")
st.title("🌿 Hybrid CNN + LLM for Leaf Disease Diagnosis")
st.caption("Mirror of the notebook structure with a Streamlit UI. EfficientNetV2-M for classification + FLAN-T5 for explanation.")

with st.sidebar:
    st.header("⚙️ Settings")
    default_model_path = "best_model_overall.pth"
    model_path = st.text_input("Model file path (.pth)", value=default_model_path)
    device_choice = st.selectbox("Device", ["Auto", "cpu", "cuda"], index=0)
    use_llm = st.checkbox("Enable LLM explanation", value=True)
    llm_name = st.selectbox("LLM model", ["google/flan-t5-base", "google/flan-t5-large"], index=0)
    st.markdown("---")
    st.write("**Input Image**")
    img_file = st.file_uploader("Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    img_url = st.text_input("Or image URL", value="")
    run_btn = st.button("🚀 Run Prediction")

with st.expander("📚 Class List (must match training order)"):
    st.code("\n".join([f"- {i+1}. {c}" for i, c in enumerate(class_names)]))

col1, col2 = st.columns([1, 1])

if run_btn:
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model file not found: '{model_path}'")
    elif not img_file and not img_url:
        st.error("Please upload an image or provide a URL.")
    else:
        dev = None if device_choice == "Auto" else device_choice
        with st.spinner("Loading model..."):
            model, device_used = load_model_cached(model_path, device_str=dev)

        llm_pipe = None
        if use_llm:
            with st.spinner("Loading LLM..."):
                llm_pipe = load_llm_cached(llm_name)

        # Determine image reference compatible with test_custom_image
        if img_file is not None:
            tmp_path = "uploaded_image.png"
            with open(tmp_path, "wb") as f:
                f.write(img_file.read())
            image_ref = tmp_path
        else:
            image_ref = img_url

        with st.spinner("Running inference..."):
            pred, conf, img, expl = predict_and_explain_streamlit(
                image_ref, model, transform_val, class_names, device_used, disease_info, llm_pipe
            )

        if pred is None:
            st.error("Prediction failed.")
        else:
            with col1:
                st.subheader("🎯 Prediction")
                st.metric("Top-1 Prediction", pred, f"{conf*100:.2f}%")
                st.subheader("🧠 Explanation")
                st.write(expl)

            with col2:
                st.subheader("🖼️ Input Image")
                if img is not None:
                    st.image(img, caption=f"Prediction: {pred} ({conf*100:.2f}%)", use_container_width=True)
                else:
                    st.info("No image to display.")

st.markdown("---")

