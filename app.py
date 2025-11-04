# app.py

# --- Import All Libraries Here ---
import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
import numpy as np
from transformers import pipeline
import json
import os
from io import BytesIO
from torchvision import transforms # Move import here
import torch.nn.functional as F 

# --- Basic Configuration ---
st.set_page_config(
    page_title="Agricultural Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Disease Information ---
@st.cache_data
def load_disease_info():
    with open('disease_info.json', 'r') as f:
        return json.load(f)

disease_info = load_disease_info()

# --- Load Classification Model ---
@st.cache_resource
def load_classification_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'best_model_overall.pth' # Adjust path if different

    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the application directory.")
        st.stop()

    try:
        # 1. Create the base model architecture (EfficientNetV2-M)
        base_model = timm.create_model('efficientnetv2_rw_m', pretrained=False)

        # 2. Get the number of input features for the default classifier
        num_features = base_model.classifier.in_features

        # 3. CRITICAL: Recreate the EXACT classifier architecture used during training
        # Based on the llm_testing.txt: Sequential(Dropout, Linear, ReLU, Dropout, Linear)
        num_classes = 11 # IMPORTANT: This must match the number of classes used during training
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 128), # First Linear layer
            nn.ReLU(),                    # Activation function
            nn.Dropout(0.2),              # Second Dropout
            nn.Linear(128, num_classes)   # Final Linear layer to number of classes (11)
        )

        # 4. Assign the modified base_model to model variable
        model = base_model.to(device)

        # 5. Load the saved state dictionary
        # Ensure map_location is appropriate
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Set the model to evaluation mode
        st.success("Classification model loaded successfully.")
        return model, device

    except RuntimeError as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.error("Most likely, the model architecture in the code does not match the state_dict in the model file. Please check the number of classes and the model architecture again.")
        st.stop() # Stop the app if model fails to load
    except Exception as e:
        st.error(f"A general error occurred while loading the model: {e}")
        st.stop() # Stop the app if another error occurs

model, device = load_classification_model() # Call the function after definition

# --- Simulate LLM ---
def simulate_llm_explanation(predicted_label, disease_info_map, confidence):
    # Simulate explanation based on disease information
    info = disease_info_map.get(predicted_label, {})
    symptoms = ", ".join(info.get("general_symptoms", ["Symptoms not found"]))
    causes = info.get("cause", "Cause not found")
    treatment = info.get("treatment", "Treatment not found")
    prevention = info.get("prevention", "Prevention not found")

    explanation = f"""
    Automated Explanation (Simulation):
    - Prediction: {predicted_label} (Confidence: {confidence:.2f}%)
    - General Symptoms: {symptoms}
    - Cause: {causes}
    - Treatment: {treatment}
    - Prevention: {prevention}
    """
    return explanation.strip()

# --- Prediction Function ---
def predict_image(image, model, device):
    # Use transforms imported above
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Size used during training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        # outputs now comes from the Sequential classifier: Linear(128, num_classes)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    # IMPORTANT: Update class_names according to the number of classes used during training (11)
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
        # Harus 11 kelas sesuai num_classes = 11 di atas
    ]

    # Ensure the predicted index does not exceed the number of defined class_names
    if predicted_idx.item() >= len(class_names):
         st.error(f"Prediction index ({predicted_idx.item()}) exceeds the number of defined classes ({len(class_names)}).")
         st.stop()

    predicted_label = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    return predicted_label, confidence_percent


# --- Streamlit Interface ---
st.title("🌿 Agricultural Disease Classifier")
st.markdown("Upload a leaf image of apple, grape, or potato to identify diseases.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Image
    image = Image.open(uploaded_file).convert('RGB') # Convert to RGB for safety
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # 2. Predict
    with st.spinner("Analyzing image..."):
        predicted_label, confidence = predict_image(image, model, device)

    with col2:
        st.subheader("Prediction Result")
        st.success(f"**Label:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

    # 3. Display Disease Information
    st.subheader("Disease Information")
    info = disease_info.get(predicted_label, {})
    if info:
        st.write(f"**General Symptoms:**")
        for symptom in info.get("general_symptoms", ["Not found"]):
            st.write(f"- {symptom}")

        st.write(f"**Cause:** {info.get('cause', 'Not found')}")
        st.write(f"**Treatment:** {info.get('treatment', 'Not found')}")
        st.write(f"**Prevention:** {info.get('prevention', 'Not found')}")
    else:
        st.warning(f"Detailed information for '{predicted_label}' was not found in the database.")

    # 4. Generate LLM Explanation (Simulation)
    st.subheader("Automated Explanation (Simulation)")
    # Simulation because large LLMs can crash Streamlit Community Cloud
    simulated_explanation = simulate_llm_explanation(predicted_label, disease_info, confidence)
    st.write(simulated_explanation)


# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, PyTorch, and Hugging Face Transformers.")
