# app.py
# =============================================================
# Streamlit App converted from: tftf.ipynb
# Notes:
# - This code is refactored to run as a Streamlit app.
# - All comments and UI strings are in English per your request.
# - Ensure the model file 'best_model_overall.pth' is present in the working directory.
# =============================================================

import os
from io import BytesIO
import traceback
import requests
import streamlit as st

import torch
import torch.nn as nn
from PIL import Image
import timm
from timm.data import create_transform

# ---------------------------------------------------------
# 1) Streamlit page configuration
# ---------------------------------------------------------
st.set_page_config(page_title="CNN + LLM Explainability", page_icon="🌿", layout="wide")

# ---------------------------------------------------------
# 2) Disease knowledge mapping (from the notebook)
# ---------------------------------------------------------
# --- Disease Knowledge Mapping
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
            "Leaves show chlorotic (yellow) spots with tiger-stripe banding",
            "Berries may shrivel or display sunburn-like symptoms",
            "Associated with fungal trunk disease (wood infection)"
        ],
        "distinguishing_features": [
            "Characteristic 'tiger-stripe' pattern on leaves",
            "Occurs more often in older vines"
        ],
        "early_actions": [
            "Manage trunk wounds to prevent infection",
            "Sanitation pruning; plan long-term trunk disease management"
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
            "No tiger-stripe pattern as in Esca"
        ],
        "early_actions": [
            "Sanitize fallen leaves and improve canopy airflow",
            "Pruning and canopy management to reduce inoculum"
        ]
    },
    "Grape___healthy": {
        "general_symptoms": [
            "Uniform green leaves without necrotic spots",
            "No fungal growth or pustules",
            "No concentric patterns or elongated lesions"
        ],
        "distinguishing_features": [
            "Absence of all disease-specific cues"
        ],
        "early_actions": [
            "Continue good cultural practices",
            "Monitor regularly for early detection"
        ]
    },
    "Apple___Apple_scab": {
        "general_symptoms": [
            "Olive-gray to brown scabby lesions on leaves and fruit with rough texture",
            "Lesions can coalesce into larger patches",
            "Fruit may become deformed"
        ],
        "distinguishing_features": [
            "Hard, rough, olive-gray/brown scabs",
            "Often appears on young leaves and fruit"
        ],
        "early_actions": [
            "Sanitize fallen leaves",
            "Use resistant cultivars and fungicides if needed"
        ]
    },
    "Apple___Black_rot": {
        "general_symptoms": [
            "Large brown necrotic leaf spots with target-like rings",
            "Fruit develops firm brown rot with a 'frog-eye' appearance",
            "Branches may show elongated cankers"
        ],
        "distinguishing_features": [
            "Target pattern on leaves",
            "Frog-eye rot pattern on fruit"
        ],
        "early_actions": [
            "Sanitize: remove infected fruit and leaves",
            "Apply preventive fungicide"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "general_symptoms": [
            "Yellow spots on upper leaf surface; orange umbrella-like aecia beneath",
            "Heavy infections can cause defoliation",
            "Fruit may develop firm lesions"
        ],
        "distinguishing_features": [
            "Orange, umbrella-like aecia on the underside of leaves",
            "Requires juniper as an alternate host"
        ],
        "early_actions": [
            "Avoid planting apples near junipers",
            "Use fungicides if necessary"
        ]
    },
    "Apple___healthy": {
        "general_symptoms": [
            "Uniform green leaves without necrotic spots",
            "No fungal growth or pustules",
            "No concentric patterns or elongated lesions"
        ],
        "distinguishing_features": [
            "Absence of all disease-specific cues"
        ],
        "early_actions": [
            "Continue good cultural practices",
            "Monitor regularly for early detection"
        ]
    },
    "Potato___Early_blight": {
        "general_symptoms": [
            "Brown leaf spots with clear concentric rings, starting on older leaves",
            "Chlorotic (yellow) halo around spots",
            "Often appears early in the growing season"
        ],
        "distinguishing_features": [
            "Well-defined concentric ring pattern",
            "Frequently begins on older foliage first"
        ],
        "early_actions": [
            "Remove infected leaves and maintain plant spacing",
            "Manage nutrition to minimize plant stress"
        ]
    },
    "Potato___Late_blight": {
        "general_symptoms": [
            "Water-soaked grayish lesions on leaves that expand rapidly",
            "White fungal growth on the undersides of leaves in humid conditions",
            "Can infect stems and tubers; progresses very quickly"
        ],
        "distinguishing_features": [
            "Very rapid, water-soaked lesion expansion",
            "White mycelium along leaf undersides when humid"
        ],
        "early_actions": [
            "Remove infected tissue; follow local fungicide recommendations if needed",
            "Avoid prolonged leaf wetness and standing water"
        ]
    },
    "Potato___healthy": {
        "general_symptoms": [
            "Uniform green leaves without necrotic spots",
            "No fungal growth or pustules",
            "No concentric rings or elongated lesions"
        ],
        "distinguishing_features": [
            "Absence of all disease-specific cues"
        ],
        "early_actions": [
            "Continue good cultural practices",
            "Monitor regularly for early detection"
        ]
    }
}

# ---------------------------------------------------------
# 3) Validation transform and class names (from the notebook)
# ---------------------------------------------------------
# --- Recreate the validation transform (assuming data_config from training) ---
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

# Define class names (must match the order used during training)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
]

# ---------------------------------------------------------
# 4) Cached loaders: CNN model and LLM
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cnn_model(model_path: str, device: str | None = None):
    """Load EfficientNetV2-M + custom classifier head, return eval-mode model and device."""
    try:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev_t = torch.device(dev)

        # Recreate the architecture exactly as in training
        base_model = timm.create_model('efficientnetv2_rw_m', pretrained=False)
        num_features = base_model.classifier.in_features
        num_classes = 11  # must match training
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        model = base_model.to(dev_t)
        state = torch.load(model_path, map_location=dev_t)
        model.load_state_dict(state)
        model.eval()
        return model, dev_t
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{model_path}' -> {e}")


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str = "google/flan-t5-base"):
    """Load a text2text-generation pipeline from HuggingFace (FLAN-T5)."""
    try:
        from transformers import pipeline
        pipe = pipeline(
            "text2text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )
        return pipe
    except Exception as e:
        raise RuntimeError(f"Failed to load LLM '{model_name}' -> {e}")


# ---------------------------------------------------------
# 5) Utilities
# ---------------------------------------------------------
def load_image_from_input(uploaded_file=None, url: str = ""):
    """Load image from Streamlit uploader or HTTP URL."""
    try:
        if uploaded_file is not None:
            return Image.open(uploaded_file).convert("RGB")
        if url.strip():
            resp = requests.get(url.strip(), timeout=20)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}" )
    return None


def get_topk(model, input_tensor, k: int = 3):
    """Compute probabilities and return top-k indices and probabilities."""
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        topk_prob, topk_idx = torch.topk(probs, k)
        return topk_idx.cpu().tolist(), topk_prob.cpu().tolist()


def build_llm_prompt(label: str, disease_info_map: dict) -> str:
    """Create a concise, structured prompt for LLM based on predicted label and knowledge mapping."""
    info = disease_info_map.get(label, {})
    general = info.get("general_symptoms", [])
    distinct = info.get("distinguishing_features", [])
    actions = info.get("early_actions", [])
    prompt = f"""
    You are an agronomy assistant who explains plant leaf disease classification results **clearly, informatively, and accurately** in English. Ground your explanation in agronomic knowledge and visible symptoms.
    ### CNN Inference Results
    - Primary predicted label: **{predicted_label}**
    - Model confidence (probability): **{confidence:.3f}**
    ### Characteristic Symptoms (from internal knowledge base)
    - General symptoms: {('; '.join(general_symptoms))}
    - Distinguishing features: {('; '.join(distinguishing_features))}
    ### Your Tasks
    1. Explain **why** this image most likely belongs to the label '{predicted_label}', linking your reasoning to the characteristic symptoms provided.
    2. Highlight **distinguishing cues** that differentiate this disease from others (e.g., 'Early_blight has concentric rings, unlike Late_blight which is water-soaked').
    3. Provide **safe, general early actions** for follow-up (no brand-specific fungicide prescriptions).
    4. Use a professional, concise tone; maximum **8–10 sentences**.
    5. Do not invent facts beyond the known symptom domain; if uncertain, **state the uncertainty briefly**.
    """
    return prompt


def generate_explanation(llm_pipe, label: str, disease_info_map: dict) -> str:
    """Call the LLM to generate an explanation using our structured prompt."""
    try:
        prompt = build_llm_prompt(label, disease_info_map)
        result = llm_pipe(prompt, num_return_sequences=1)
        return result[0]['generated_text']
    except Exception as e:
        return f"Automatic explanation is not available at the moment ({e})."


# ---------------------------------------------------------
# 6) Streamlit UI
# ---------------------------------------------------------
st.title("🌿 Hybrid CNN + LLM for Leaf Disease Diagnosis")
st.caption("EfficientNetV2-M (PyTorch) for classification + FLAN-T5 for explainability.")

with st.sidebar:
    st.header("⚙️ Settings")
    default_model_path = "best_model_overall.pth"
    model_path = st.text_input("Model file path (.pth)", value=default_model_path, help="Make sure the file exists in the working folder.")
    force_device = st.selectbox("Select device (optional)", options=["Auto", "cpu", "cuda"], index=0)
    use_llm = st.checkbox("Enable LLM explanation", value=True)
    llm_name = st.selectbox("LLM model", ["google/flan-t5-base", "google/flan-t5-large"], index=0, help="Choose a variant that fits memory.")
    st.markdown("---")
    st.write("**Input Image**")
    img_file = st.file_uploader("Upload leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    img_url = st.text_input("Or image URL", value="")

    run_btn = st.button("🚀 Run Prediction")


with st.expander("📚 Class List (must match training order)"):
    st.code("\n".join([f"- {i+1}. {c}" for i, c in enumerate(class_names)]))

result_col, img_col = st.columns([1, 1])

if run_btn:
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model file not found: '{model_path}'. Upload it to the working directory or adjust the path.")
    elif not img_file and not img_url:
        st.error("Please upload an image or provide an image URL first.")
    else:
        dev = None if force_device == "Auto" else force_device

        with st.spinner("Loading model..."):
            model, dev_t = load_cnn_model(model_path, device=dev)

        llm_pipe = None
        if use_llm:
            with st.spinner("Loading LLM..."):
                llm_pipe = load_llm(llm_name)

        try:
            image = load_image_from_input(img_file, img_url)
        except Exception as e:
            st.error(str(e))
            image = None

        if image is not None:
            tensor = transform_val(image).unsqueeze(0).to(dev_t)

            with st.spinner("Running inference..."):
                top_idx, top_prob = get_topk(model, tensor, k=3)
                top_labels = [class_names[i] for i in top_idx]
                top_percent = [float(p) * 100 for p in top_prob]

            with result_col:
                st.subheader("🎯 Prediction Result")
                st.metric("Top-1 Prediction", top_labels[0], f"{top_percent[0]:.2f}%" )
                st.write("Top-3 predictions:")
                for lbl, pr in zip(top_labels, top_percent):
                    st.write(f"- **{lbl}**: {pr:.2f}%" )

                if use_llm and llm_pipe is not None:
                    st.subheader("🧠 Explanation (LLM)")
                    explanation = generate_explanation(llm_pipe, top_labels[0], disease_info)
                    st.write(explanation)
                else:
                    st.info("LLM explanation is disabled. Enable it in the sidebar to see the reasoning.")

            with img_col:
                st.subheader("🖼️ Input Image")
                st.image(image, caption=f"Input - Prediction: {top_labels[0]} ({top_percent[0]:.2f}%)", use_container_width=True)

st.markdown("---")

