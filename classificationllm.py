# tftf.py
# ============================================================================
# Modul util tanpa side-effect untuk:
# - konstanta kelas & info penyakit
# - build/load model EfficientNetV2-M (timm)
# - transform validasi
# - inferensi (predict_image)
# - load LLM HF pipeline (dengan fallback offline)
# - generate_explanation
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline
import timm
from timm.data import create_transform

# =========================
# Konstanta dataset/penyakit
# =========================
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
]

DISEASE_INFO = {
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
        "distinguishing_features": ["Absence of all disease-specific cues"],
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
        "distinguishing_features": ["Absence of all disease-specific cues"],
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
        "distinguishing_features": ["Absence of all disease-specific cues"],
        "early_actions": [
            "Continue good cultural practices",
            "Monitor regularly for early detection"
        ]
    }
}

# =========================
# Util model & transform
# =========================
def get_device():
    """Pilih CUDA jika tersedia, selain itu CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(num_classes: int = 11):
    """Bangun arsitektur EfficientNetV2-M dengan head classifier sesuai training."""
    base = timm.create_model('efficientnetv2_rw_m', pretrained=False)
    in_feats = base.classifier.in_features
    base.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    return base

def load_model(weights_path: str = "best_model_overall.pth", device=None):
    """
    Muat model + bobot terlatih.
    Pastikan head classifier cocok dengan saat training.
    """
    if device is None:
        device = get_device()
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)  # jika head berbeda, sesuaikan arsitektur
    model.eval()
    return model, device

def get_transform():
    """Transform validasi konsisten dengan training/preprocess timm."""
    data_config = {
        'input_size': (3, 320, 320),
        'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225),
        'crop_pct': 1.0,
        'crop_mode': 'center'
    }
    return create_transform(**data_config, is_training=False)

# =========================
# Inference + LLM
# =========================
def predict_image(image: Image.Image, model, transform, device):
    """
    Lakukan prediksi 1 gambar.
    Return: (predicted_label: str, confidence: float [0..1], probs: 1D tensor len=11 (CPU))
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        idx = int(torch.argmax(probs).item())
    return CLASS_NAMES[idx], float(probs[idx].item()), probs

def load_llm():
    """
    Buat HF text2text pipeline (FLAN-T5). Jika gagal (mis. offline),
    kembalikan fallback object yang punya __call__ agar antarmuka tetap sama.
    """
    try:
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=0 if torch.cuda.is_available() else -1,
            max_length=200
        )
    except Exception:
        class _FallbackLLM:
            def __call__(self, prompt, **kwargs):
                return [{
                    "generated_text": (
                        "Explanation unavailable from LLM (offline/failure). "
                        "Please rely on characteristic symptoms and distinguishing features for now."
                    )
                }]
        return _FallbackLLM()

def generate_explanation(predicted_label: str, confidence: float, llm_pipeline):
    """
    Bangkitkan penjelasan berbasis LLM (dengan prompt terstruktur).
    Jika kelas tidak terdaftar di DISEASE_INFO, berikan pesan generik.
    """
    if predicted_label not in DISEASE_INFO:
        return (
            f"The model predicted '{predicted_label}' with confidence {confidence:.3f}. "
            f"Detailed information for this specific class is not available."
        )

    det = DISEASE_INFO[predicted_label]
    general_symptoms = det.get("general_symptoms", ["Symptoms not specified."])
    distinguishing = det.get("distinguishing_features", ["Features not specified."])

    prompt = f"""
You are an agronomy assistant who explains plant leaf disease classification results clearly and accurately in English.
### CNN Inference Results
- Primary predicted label: **{predicted_label}**
- Model confidence (probability): **{confidence:.3f}**
### Characteristic Symptoms
- General symptoms: {'; '.join(general_symptoms)}
- Distinguishing features: {'; '.join(distinguishing)}
### Your Tasks
1. Explain why this image likely belongs to '{predicted_label}', linking to characteristic symptoms.
2. Highlight distinguishing cues that differentiate this disease from others.
3. Provide safe, general early actions for follow-up.
4. Use a professional, concise tone; maximum 8–10 sentences.
"""
    out = llm_pipeline(prompt, max_length=200, do_sample=True, temperature=0.7, truncation=True)
    return out[0]['generated_text']
