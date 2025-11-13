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
# 6. Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="CNN + LLM Plant Disease Classifier",
        page_icon="🌿",
        layout="wide",
    )

    st.title("🌿 Hybrid CNN + LLM for Plant Leaf Disease Classification")
    st.write(
        "Upload a leaf image atau paste image URL untuk mendapatkan label penyakit "
        "dan penjelasan tekstual dari LLM."
    )

    # Sidebar: status model & LLM
    with st.sidebar:
        st.header("Model status")

        try:
            model = load_model()
            transform = load_transform()
            st.success("CNN model loaded ✅")
        except Exception as e:
            st.error(f"Error loading CNN model: {e}")
            st.stop()

        try:
            llm = load_llm()
            st.success("LLM ready (Replicate) ✅")
        except Exception as e:
            st.warning(f"LLM not available: {e}")
            llm = None

    uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    url = st.text_input("Or input image URL")

    if st.button("Diagnose"):
        image, source = load_image_from_input(uploaded, url)
        if image is None:
            st.error("Please upload an image or provide a URL first.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption=f"Input image ({source})", use_container_width=True)

        with st.spinner("Running CNN prediction..."):
            # Ambil lagi dari cache_resource (instansinya sama)
            model = load_model()
            transform = load_transform()
            label, confidence = predict(image, model, transform)

        with col2:
            st.subheader("Prediction")
            st.markdown(f"**Label:** `{label}`")
            st.markdown(f"**Confidence:** {confidence:.2%}")

        # Penjelasan LLM (jika tersedia)
        if llm is not None:
            with st.spinner("Generating explanation with LLM..."):
                explanation = generate_explanation(llm, label)
            st.subheader("LLM Explanation")
            st.write(explanation)
        else:
            st.info(
                "LLM belum dikonfigurasi, jadi penjelasan teks tidak tersedia. "
                "Set REPLICATE_API_TOKEN untuk mengaktifkan fitur ini."
            )


if __name__ == "__main__":
    main()

