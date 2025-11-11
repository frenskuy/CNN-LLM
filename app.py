import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st
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

def download_model_from_url(url, save_path='best_model_overall.pth'):
    """Download model from URL if not exists locally"""
    import os
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        if file_size > 1000:  # File exists and is not a placeholder
            return True
    
    try:
        st.info(f"Downloading model from URL...")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                progress_bar = st.progress(0)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress_bar.progress(min(downloaded / total_size, 1.0))
        
        st.success("✅ Model downloaded successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to download model: {e}")
        return False

@st.cache_resource
def load_model_from_file(model_path):
    """Load model from specific file path"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import os
    
    if not os.path.exists(model_path):
        st.error(f"Model file {model_path} not found!")
        return None, None
    
    file_size = os.path.getsize(model_path)
    
    if file_size < 1000:
        st.error(f"⚠️ File {model_path} is too small ({file_size} bytes). This is likely a Git LFS pointer file.")
        st.info("""
        **To fix this, run in your local repository:**
        ```bash
        git lfs pull
        git add .
        git commit -m "Pull LFS files"
        git push
        ```
        """)
        return None, None
    
    try:
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
        
        with st.spinner(f"Loading weights from {model_path}..."):
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e1:
                st.warning(f"Trying alternative loading method...")
                try:
                    state_dict = torch.load(model_path, map_location=device, weights_only=False)
                    model.load_state_dict(state_dict)
                except Exception as e2:
                    st.error(f"Failed to load: {e2}")
                    raise e2
        
        model.eval()
        st.success(f"✅ Model loaded from {model_path}!")
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

@st.cache_resource
def load_model(model_url=None):
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import os
    
    # Try multiple possible model file names
    possible_paths = [
        'best_model_overall.pth',
        './best_model_overall.pth',
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            if file_size > 1000:  # Not a Git LFS pointer
                model_path = path
                st.info(f"✅ Found model file: {path}")
                break
    
    # Try to download from URL if provided and file doesn't exist
    if model_url and model_path is None:
        model_path = 'best_model_overall.pth'
        if not download_model_from_url(model_url, model_path):
            return None, None
    
    # Check if model file exists
    if model_path is None or not os.path.exists(model_path):
        st.error(f"❌ Model file not found in repository.")
        
        # Show all files in current directory for debugging
        st.warning("📂 Files found in current directory:")
        files = os.listdir('.')
        for f in files:
            size = os.path.getsize(f) if os.path.isfile(f) else 0
            st.text(f"  - {f} ({size} bytes)")
        
        st.info("""
        **Please provide model file using one of these methods:**
        
        1. **Upload via GitHub with Git LFS** (for files > 100MB):
           ```bash
           git lfs install
           git lfs track "*.pth"
           git add .gitattributes best_model_overall.pth
           git commit -m "Add model"
           git push
           ```
        
        2. **Host on Google Drive:**
           - Upload your .pth file to Google Drive
           - Get shareable link (Anyone with the link can view)
           - Convert to direct download link
           - Add URL to secrets.toml or code
        
        3. **Use Hugging Face Hub:**
           - Upload model to Hugging Face
           - Download in app using URL
        """)
        return None, None
    
    # Check file size
    file_size = os.path.getsize(model_path)
    st.info(f"📦 Model file found. Size: {file_size / (1024*1024):.2f} MB")
    
    if file_size < 1000:  # Less than 1KB - likely a Git LFS pointer
        st.error("⚠️ Model file appears to be a Git LFS pointer file (too small).")
        st.info("""
        **Fix Git LFS issue:**
        ```bash
        git lfs pull
        git add best_model_overall.pth
        git commit -m "Update model file"
        git push
        ```
        Or provide a direct download URL for the model.
        """)
        return None, None
    
    try:
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
        
        with st.spinner("Loading model weights..."):
            try:
                # Try normal loading
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
            except Exception as e1:
                st.warning(f"Normal loading failed, trying alternative method...")
                try:
                    # Try loading with weights_only=False
                    state_dict = torch.load(model_path, map_location=device, weights_only=False)
                    model.load_state_dict(state_dict)
                except Exception as e2:
                    raise e2
        
        model.eval()
        st.success("✅ Model loaded successfully!")
        
        return model, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **The model file might be corrupted. Please:**
        1. Re-download the original model file
        2. Verify the file is not corrupted (check file size)
        3. Re-upload to GitHub using Git LFS
        """)
        return None, None

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

def generate_explanation(predicted_label, confidence):
    """Generate structured explanation from disease info"""
    if predicted_label not in disease_info:
        return f"The model predicted '{predicted_label}' with confidence {confidence:.1%}. Detailed information for this specific class is not available."
    
    disease_details = disease_info[predicted_label]
    general_symptoms = disease_details.get("general_symptoms", ["Symptoms not specified."])
    distinguishing_features = disease_details.get("distinguishing_features", ["Features not specified."])
    early_actions = disease_details.get("early_actions", ["General monitoring recommended."])
    
    # Parse the predicted label
    parts = predicted_label.split('___')
    plant_type = parts[0] if len(parts) > 0 else "Plant"
    disease_name = parts[1].replace('_', ' ') if len(parts) > 1 else "Unknown"
    
    # Create explanation based on whether it's healthy or diseased
    if "healthy" in disease_name.lower():
        explanation = f"### 🌿 Diagnosis: Healthy {plant_type} Plant\n\n"
        explanation += f"The analysis indicates that this {plant_type.lower()} leaf appears **healthy** with a confidence level of **{confidence:.1%}**. "
        explanation += f"The leaf exhibits {general_symptoms[0].lower()}, which are characteristic indicators of healthy plant tissue. "
        explanation += "The absence of disease symptoms such as lesions, discoloration patterns, or fungal growth confirms this positive assessment.\n\n"
        
        explanation += "**Visual Characteristics:**\n\n"
        explanation += f"The examined leaf shows {distinguishing_features[0].lower()}, which is typical of well-maintained and disease-free foliage. "
        explanation += "The uniform coloration and intact tissue structure indicate proper nutrient uptake and absence of pathogen infection.\n\n"
        
        explanation += "**Recommendations:**\n\n"
        for i, action in enumerate(early_actions, 1):
            explanation += f"{i}. {action}\n"
        explanation += f"{len(early_actions) + 1}. Maintain current cultivation practices including proper watering, fertilization, and pest management\n"
        explanation += f"{len(early_actions) + 2}. Regular monitoring is essential to detect any early signs of disease development\n"
    else:
        explanation = f"### 🔬 Diagnosis: {disease_name.title()}\n\n"
        explanation += f"The image analysis has identified **{disease_name.lower()}** affecting this {plant_type.lower()} plant with a confidence level of **{confidence:.1%}**. "
        explanation += f"This disease typically manifests as {general_symptoms[0].lower()}. "
        
        if len(general_symptoms) > 1:
            explanation += f"Additionally, affected plants may exhibit {general_symptoms[1].lower()}. "
        
        if len(general_symptoms) > 2:
            explanation += f"{general_symptoms[2]} "
        
        explanation += "\n\n**Key Distinguishing Features:**\n\n"
        for i, feature in enumerate(distinguishing_features, 1):
            explanation += f"{i}. {feature}\n"
        
        explanation += f"\nThese specific visual indicators help differentiate {disease_name.lower()} from other common plant diseases, ensuring accurate diagnosis and appropriate treatment planning.\n\n"
        
        explanation += "**Recommended Actions:**\n\n"
        for i, action in enumerate(early_actions, 1):
            explanation += f"{i}. {action}\n"
        
        explanation += "\n**Important Note:** Early detection and proper disease management are crucial for preventing disease spread and minimizing crop damage. "
        explanation += "Consider consulting with a local agricultural extension officer for region-specific treatment recommendations and best practices.\n"
    
    return explanation

# Main app
def main():
    st.title("🌿 Plant Disease Classification System")
    st.write("Upload an image of a plant leaf to detect diseases in Apple, Grape, and Potato plants.")
    
    import os
    
    # Show model file selector in sidebar
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # List all .pth files in directory
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        
        if pth_files:
            st.success(f"Found {len(pth_files)} model file(s):")
            selected_model = st.selectbox("Select model file:", pth_files)
            
            # Show file info
            if selected_model:
                file_size = os.path.getsize(selected_model)
                st.info(f"Size: {file_size / (1024*1024):.2f} MB")
                
                if file_size < 1000:
                    st.warning("⚠️ File seems too small. Might be Git LFS pointer.")
        else:
            st.warning("No .pth files found in repository")
            selected_model = None
        
        # Manual URL input
        st.subheader("Or provide URL:")
        manual_url = st.text_input(
            "Model URL", 
            placeholder="https://...",
            help="Direct download URL for model file"
        )
    
    model_url = manual_url if manual_url else None
    
    # Load model with selected file
    with st.spinner("Loading model..."):
        if pth_files and selected_model:
            # Override the load_model to use selected file
            model, device = load_model_from_file(selected_model)
        else:
            model, device = load_model(model_url)
        
    if model is None:
        st.error("Failed to load model. Please check the configuration.")
        return
        
    transform = get_transform()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses a deep learning model (EfficientNetV2-M) "
        "to classify plant diseases in Apple, Grape, and Potato leaves. "
        "It provides detailed explanations based on agronomic knowledge."
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
                response = requests.get(image_url, timeout=10)
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
                explanation = generate_explanation(predicted_label, confidence)
                st.markdown(explanation)
                
                # Display disease information in expandable sections
                if predicted_label in disease_info:
                    st.subheader("🔬 Additional Disease Information")
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
