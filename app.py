# app.py

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

# --- Konfigurasi Dasar ---
st.set_page_config(
    page_title="Agricultural Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Informasi Penyakit ---
@st.cache_data
def load_disease_info():
    with open('disease_info.json', 'r') as f:
        return json.load(f)

disease_info = load_disease_info()

# --- Load Model Klasifikasi ---
@st.cache_resource
def load_classification_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'best_model_overall.pth' # Sesuaikan path jika berbeda

    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' tidak ditemukan. Harap pastikan file model ada di direktori aplikasi.")
        st.stop()

    try:
        # Buat arsitektur model sesuai training
        # Penting: Buat model dengan classifier default terlebih dahulu
        model = timm.create_model('efficientnetv2_rw_m', pretrained=False)

        # Dapatkan jumlah fitur dari classifier default
        num_features = model.classifier.in_features

        # Ganti layer classifier dengan yang sesuai jumlah kelas (12)
        model.classifier = nn.Linear(num_features, 12)

        # Muat state_dict ke model yang sudah dimodifikasi arsitekturnya
        # Pastikan map_location sesuai
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        st.success("Model klasifikasi berhasil dimuat.")
        return model, device

    except RuntimeError as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.error("Kemungkinan besar arsitektur model di kode tidak cocok dengan state_dict dalam file model. Periksa kembali jumlah kelas dan arsitektur model.")
        st.stop() # Hentikan aplikasi jika model gagal dimuat
    except Exception as e:
        st.error(f"Terjadi kesalahan umum saat memuat model: {e}")
        st.stop() # Hentikan aplikasi jika terjadi error lain
        
# --- Load LLM ---
@st.cache_resource
def load_llm():
    # Gunakan GPU jika tersedia, jika tidak gunakan CPU
    device_num = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-large", # Anda bisa mengganti ini dengan model lain jika diinginkan
        device=device_num, # Gunakan -1 untuk CPU
        max_length=200,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32 # Opsional: hemat memori GPU
    )
    return pipe

# Memuat LLM bisa memakan waktu lama dan memori, jadi hati-hati
# Untuk demo, kita bisa gunakan model yang lebih kecil atau simulasikan
# st.info("Memuat LLM (ini bisa memakan waktu beberapa menit)...") # Beri indikasi
# llm_pipe = load_llm()
# st.success("LLM siap!")

# --- Simulasi LLM ---
def simulate_llm_explanation(predicted_label, disease_info_map, confidence):
    # Simulasi penjelasan berdasarkan informasi penyakit
    info = disease_info_map.get(predicted_label, {})
    symptoms = ", ".join(info.get("general_symptoms", ["Gejala tidak ditemukan"]))
    causes = info.get("cause", "Penyebab tidak ditemukan")
    treatment = info.get("treatment", "Perawatan tidak ditemukan")
    prevention = info.get("prevention", "Pencegahan tidak ditemukan")

    explanation = f"""
    Penjelasan Otomatis (Simulasi):
    - Prediksi: {predicted_label} (Confidence: {confidence:.2f}%)
    - Gejala Umum: {symptoms}
    - Penyebab: {causes}
    - Perawatan: {treatment}
    - Pencegahan: {prevention}
    """
    return explanation.strip()

# --- Fungsi Prediksi ---
def predict_image(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Ukuran yang digunakan saat training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'
    ]

    predicted_label = class_names[predicted_idx.item()]
    confidence_percent = confidence.item() * 100

    return predicted_label, confidence_percent

# --- Import Transform di sini ---
from torchvision import transforms

# --- Antarmuka Streamlit ---
st.title("🌿 Klasifikasi Penyakit Tanaman")
st.markdown("Unggah gambar daun tanaman apel, anggur, atau kentang untuk mengidentifikasi penyakitnya.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan Gambar
    image = Image.open(uploaded_file).convert('RGB') # Konversi ke RGB untuk keamanan
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Gambar yang Diunggah', use_column_width=True)

    # 2. Prediksi
    with st.spinner("Menganalisis gambar..."):
        predicted_label, confidence = predict_image(image, model, device)

    with col2:
        st.subheader("Hasil Prediksi")
        st.success(f"**Label:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

    # 3. Tampilkan Informasi Penyakit
    st.subheader("Informasi Penyakit")
    info = disease_info.get(predicted_label, {})
    if info:
        st.write(f"**Gejala Umum:**")
        for symptom in info.get("general_symptoms", ["Tidak ditemukan"]):
            st.write(f"- {symptom}")

        st.write(f"**Penyebab:** {info.get('cause', 'Tidak ditemukan')}")
        st.write(f"**Perawatan:** {info.get('treatment', 'Tidak ditemukan')}")
        st.write(f"**Pencegahan:** {info.get('prevention', 'Tidak ditemukan')}")
    else:
        st.warning(f"Informasi rinci untuk '{predicted_label}' tidak ditemukan dalam database.")

    # 4. Generate Penjelasan LLM (Simulasi)
    st.subheader("Penjelasan Otomatis (Simulasi)")
    # Simulasi karena LLM besar bisa menyebabkan crash di Streamlit Community Cloud
    simulated_explanation = simulate_llm_explanation(predicted_label, disease_info, confidence)
    st.write(simulated_explanation)

    # --- Bagian Kode LLM Asli (Opsional, untuk referensi atau deployment lokal) ---
    # Uncomment kode di bawah ini dan comment blok simulasi di atas jika ingin menggunakan LLM asli.
    # PERINGATAN: Ini sangat mungkin menyebabkan timeout atau crash di Streamlit Community Cloud.
    # st.subheader("Penjelasan Otomatis dari LLM (Asli - Bisa Lama)")
    # try:
    #     prompt = f"Jelaskan penyakit tanaman {predicted_label} beserta gejala, penyebab, perawatan, dan pencegahannya dalam bahasa Indonesia yang mudah dimengerti. Gunakan informasi ini: {disease_info.get(predicted_label, {}).get('general_symptoms', [])}."
    #     with st.spinner("LLM sedang memproses..."):
    #         # Gunakan model yang lebih kecil atau atur ulang pipeline jika perlu
    #         llm_pipe = load_llm() # Muat lagi jika cache dihapus
    #         result = llm_pipe(prompt)
    #     explanation = result[0]['generated_text']
    #     st.write(explanation)
    # except Exception as e:
    #     st.error(f"Terjadi kesalahan saat menggunakan LLM: {str(e)}")
    #     st.write("Mencoba simulasi penjelasan...")
    #     simulated_explanation = simulate_llm_explanation(predicted_label, disease_info, confidence)
    #     st.write(simulated_explanation)


# --- Footer ---
st.markdown("---")

st.caption("Dibangun dengan ❤️ menggunakan Streamlit, PyTorch, dan Hugging Face Transformers.")
