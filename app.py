import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
from transformers import pipeline
import requests
from io import BytesIO
import timm
from timm.data import create_transform

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# Disease information mapping
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

# Class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
]

@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model architecture
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
    
    # Load weights
    model = base_model.to(device)
    model.load_state_dict(torch.load('best_model_overall.pth', map_location=device))
    model.eval()
    
    return model, device

@st.cache_resource
def load_llm():
    """Load the LLM pipeline"""
    llm_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=0 if torch.cuda.is_available() else -1,
        max_length=200
    )
    return llm_pipe

def get_transform():
    """Get the validation transform"""
    data_config = {
        'input_size': (3, 320, 320),
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 1.0,
        'crop_mode': 'center'
    }
    return create_transform(**data_config, is_training=False)

def predict_image(image, model, transform, device):
    """Predict the class of an image"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform and predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted_class_index = torch.max(probabilities, 1)
        
        predicted_class_name = class_names[predicted_class_index.item()]
        confidence = probabilities[0, predicted_class_index.item()].item()
        
        return predicted_class_name, confidence, probabilities[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None, None

def generate_explanation(predicted_label, confidence, llm_pipeline):
    """Generate explanation using LLM"""
    if predicted_label not in disease_info:
        return f"The model predicted '{predicted_label}' with confidence {confidence:.3f}. Detailed information for this specific class is not available."
    
    disease_details = disease_info[predicted_label]
    general_symptoms = disease_details.get("general_symptoms", ["Symptoms not specified."])
    distinguishing_features = disease_details.get("distinguishing_features", ["Features not specified."])
    
    prompt = f"""
    You are an agronomy assistant who explains plant leaf disease classification results clearly and accurately in English.
    ### CNN Inference Results
    - Primary predicted label: **{predicted_label}**
    - Model confidence (probability): **{confidence:.3f}**
    ### Characteristic Symptoms
    - General symptoms: {'; '.join(general_symptoms)}
    - Distinguishing features: {'; '.join(distinguishing_features)}
    ### Your Tasks
    1. Explain why this image likely belongs to '{predicted_label}', linking to characteristic symptoms.
    2. Highlight distinguishing cues that differentiate this disease from others.
    3. Provide safe, general early actions for follow-up.
    4. Use a professional, concise tone; maximum 8-10 sentences.
    """
    
    try:
        explanation_result = llm_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7, truncation=True)
        explanation = explanation_result[0]['generated_text']
    except Exception as e:
        st.warning(f"Could not generate LLM explanation: {e}")
        explanation = f"The model predicted '{predicted_label}' with {confidence:.1%} confidence. "
        explanation += f"General symptoms: {'; '.join(general_symptoms)}. "
        explanation += f"Distinguishing features: {'; '.join(distinguishing_features)}."
    
    return explanation

# Main app
def main():
    st.title("🌿 Plant Disease Classification System")
    st.write("Upload an image of a plant leaf to detect diseases in Apple, Grape, and Potato plants.")
    
    # Load model and LLM
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
    
    # Upload options
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
                response = requests.get(image_url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"Error loading image from URL: {e}")
    
    if image is not None:
        # Display image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
        
        # Predict button
        if st.button("🔍 Analyze Disease", type="primary"):
            with st.spinner("Analyzing image..."):
                predicted_label, confidence, all_probs = predict_image(image, model, transform, device)
            
            if predicted_label is not None:
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Display prediction
                    st.metric("Predicted Class", predicted_label.replace('___', ' - '))
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Confidence bar
                    st.progress(confidence)
                
                # Display top 3 predictions
                st.subheader("Top 3 Predictions")
                top3_probs, top3_indices = torch.topk(all_probs, 3)
                
                cols = st.columns(3)
                for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
                    with cols[i]:
                        st.metric(
                            f"#{i+1}: {class_names[idx].replace('___', ' - ')}", 
                            f"{prob.item():.2%}"
                        )
                
                # Generate and display explanation
                st.subheader("📋 Detailed Analysis")
                with st.spinner("Generating explanation..."):
                    explanation = generate_explanation(predicted_label, confidence, llm_pipe)
                
                st.write(explanation)
                
                # Display disease information
                if predicted_label in disease_info:
                    st.subheader("🔬 Disease Information")
                    disease_details = disease_info[predicted_label]
                    
                    with st.expander("General Symptoms"):
                        for symptom in disease_details["general_symptoms"]:
                            st.write(f"• {symptom}")
                    
                    with st.expander("Distinguishing Features"):
                        for feature in disease_details["distinguishing_features"]:
                            st.write(f"• {feature}")
                    
                    with st.expander("Early Actions"):
                        for action in disease_details["early_actions"]:
                            st.write(f"• {action}")

if __name__ == "__main__":
    main()
