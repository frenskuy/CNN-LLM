import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
import timm
from timm.data import create_transform
import torch.nn.functional as F
import os
import numpy as np

# --- Konfigurasi Dasar ---
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Mapping Informasi Penyakit ---
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

# --- Fungsi untuk Memuat Model ---
@st.cache_resource
def load_model_and_pipeline():
    """Loads the trained model, transformation, and LLM pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.info(f"Using device: {device}")

    # Path model Anda
    model_path = 'best_model_overall.pth' # Ganti jika path berbeda

    if not os.path.exists(model_path):
         st.error(f"Model file not found at {model_path}. Please ensure it's in the correct directory.")
         st.stop() # Hentikan eksekusi jika model tidak ditemukan

    # Re-create the model architecture
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

    model = base_model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Inisialisasi LLM
    try:
        llm_pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1,
            max_length=200,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # Opsional: kurangi ukuran memori
        )
    except Exception as e:
        st.error(f"Failed to load LLM pipeline: {e}. Proceeding without LLM explanation.")
        llm_pipe = None

    # Transformasi
    data_config = {
        'input_size': (3, 320, 320),
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 1.0,
        'crop_mode': 'center'
    }
    transform_val = create_transform(**data_config, is_training=False)

    # Nama kelas
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
    ]

    return model, transform_val, class_names, device, llm_pipe


# --- Fungsi untuk Prediksi ---
def predict_image(image, model, transform, class_names, device):
    """Prepares image and runs prediction."""
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, 1)

        predicted_class_name = class_names[predicted_class_index.item()]
        confidence_value = confidence.item()
        return predicted_class_name, confidence_value
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None


# --- Fungsi untuk Menjelaskan Prediksi ---
def explain_prediction(predicted_label, confidence, disease_info_map, llm_pipeline):
    """Generates an explanation for the prediction."""
    if predicted_label not in disease_info_map:
        return f"No detailed info found for label '{predicted_label}'."

    details = disease_info_map[predicted_label]
    general_symptoms = "; ".join(details.get("general_symptoms", ["Symptoms not specified."]))
    distinguishing_features = "; ".join(details.get("distinguishing_features", ["Features not specified."]))
    early_actions = "; ".join(details.get("early_actions", ["General monitoring recommended."]))

    if llm_pipeline is None:
        # Jika LLM tidak tersedia, gunakan informasi statis
        explanation = f"""
        **Prediction:** {predicted_label}
        **Confidence:** {confidence:.3f}

        **General Symptoms:** {general_symptoms}
        **Distinguishing Features:** {distinguishing_features}
        **Early Actions:** {early_actions}
        """
        return explanation

    prompt = f"""
    You are an agronomy assistant explaining plant leaf disease classification results clearly.
    ### CNN Inference Results
    - Primary predicted label: **{predicted_label}**
    - Model confidence: **{confidence:.3f}**
    ### Characteristic Symptoms
    - General symptoms: {general_symptoms}
    - Distinguishing features: {distinguishing_features}
    ### Your Tasks
    1. Explain why this image likely belongs to '{predicted_label}', linking to symptoms.
    2. Highlight distinguishing cues differentiating it from others.
    3. Provide safe, general early actions.
    4. Be concise, professional, 8-10 sentences max.
    5. Do not invent facts.
    """

    try:
        explanation_result = llm_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7, truncation=True)
        explanation = explanation_result[0]['generated_text']
    except Exception as e:
        st.warning(f"LLM explanation generation failed: {e}. Using static info.")
        explanation = f"""
        **Prediction:** {predicted_label}
        **Confidence:** {confidence:.3f}

        **General Symptoms:** {general_symptoms}
        **Distinguishing Features:** {distinguishing_features}
        **Early Actions:** {early_actions}
        """
    return explanation


# --- Main App ---
def main():
    st.title("🌿 Plant Disease Classifier")
    st.write("Upload an image of a plant leaf to classify its health status using AI.")

    # Load model dan pipeline saat app dijalankan
    with st.spinner("Loading model and resources..."):
        model, transform, class_names, device, llm_pipe = load_model_and_pipeline()

    # Upload file
    uploaded_file = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Tombol prediksi
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                predicted_label, confidence = predict_image(image, model, transform, class_names, device)

            if predicted_label is not None and confidence is not None:
                # Tampilkan hasil prediksi
                with col2:
                    st.subheader("Prediction Result")
                    st.success(f"**Predicted Class:** {predicted_label}")
                    st.metric(label="Confidence", value=f"{confidence:.3f}")

                # Tampilkan penjelasan
                st.subheader("Explanation")
                explanation = explain_prediction(predicted_label, confidence, disease_info, llm_pipe)
                st.write(explanation)

                # Info tambahan dari knowledge base
                if predicted_label in disease_info:
                     st.subheader("Detailed Information")
                     details = disease_info[predicted_label]
                     with st.expander("General Symptoms"):
                         st.write("; ".join(details["general_symptoms"]))
                     with st.expander("Distinguishing Features"):
                         st.write("; ".join(details["distinguishing_features"]))
                     with st.expander("Recommended Early Actions"):
                         st.write("; ".join(details["early_actions"]))

            else:
                st.error("Prediction failed. Please check the image and try again.")


if __name__ == "__main__":
    main()
