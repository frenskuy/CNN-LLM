# app.py
# =============================================================================
# Streamlit App: CNN + (Rule-based) Explainability untuk Klasifikasi Penyakit Daun
# -----------------------------------------------------------------------------
# Catatan penting:
# - Aplikasi ini DISESUAIKAN dari notebook Anda (tftf.ipynb) menjadi 1 file .py.
# - Tidak membutuhkan file lain untuk UI/logic. Namun, untuk akurasi, sangat
#   disarankan menyediakan bobot terlatih: "best_model_overall.pth"
#   di direktori kerja yang sama. Jika file bobot tidak ditemukan, model
#   akan memakai head acak (hasil prediksi tidak dapat diandalkan).
# - Komentar seluruhnya dalam Bahasa Indonesia sesuai preferensi Anda.
# - Kelas target mengikuti skema Apple, Grape, dan Potato (11 kelas).
# =============================================================================

import os
import io
import traceback
from typing import List, Tuple, Optional

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import timm
from torchvision import transforms as T
from pathlib import Path
import requests

# -----------------------------------------------------------------------------
# 1) Konfigurasi Halaman
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Classifier (CNN + Explainability)",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------------------------------------------------------
# 2) Daftar Kelas & Peta Pengetahuan Penyakit (untuk penjelasan)
# -----------------------------------------------------------------------------
# Daftar kelas (urutannya harus sama dengan saat training)
CLASS_NAMES: List[str] = [
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

# Peta pengetahuan sederhana untuk membangun penjelasan tekstual.
# (Bersifat ringkas dan non-medis; untuk edukasi awal.)
DISEASE_INFO = {
    "Grape___Black_rot": {
        "gejala_umum": [
            "Bercak cokelat pada daun dengan pola cincin konsentris (seperti target).",
            "Buah menghitam/mengeras (mummified).",
            "Lesi memanjang berwarna cokelat pada batang/cane."
        ],
        "pembeda": [
            "Pola cincin konsentris sangat khas di daun.",
            "Buah menjadi keras hitam dan kering (mumi)."
        ],
        "saran_awal": [
            "Sanitasi: buang bagian tanaman yang terinfeksi.",
            "Pertimbangkan fungisida protektif (sesuai rekomendasi setempat)."
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "gejala_umum": [
            "Daun memiliki bercak nekrotik tidak beraturan (tiger stripe).",
            "Buah menunjukkan bercak gelap seperti measles.",
            "Kadang ada gejala di kayu/vine (discoloration)."
        ],
        "pembeda": [
            "Pola garis 'tiger stripe' pada daun.",
            "Bercak gelap pada buah seperti measles."
        ],
        "saran_awal": [
            "Pangkas bagian kayu terinfeksi, tingkatkan sanitasi kebun.",
            "Kelola stres tanaman (drainase, pemangkasan tepat)."
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "gejala_umum": [
            "Bercak cokelat-hitam menyebar di daun.",
            "Daun bisa menguning dan rontok dini.",
            "Lesi cenderung tidak berpola cincin konsentris."
        ],
        "pembeda": [
            "Bercak gelap menyatu tanpa pola target ring.",
            "Kerontokan daun lebih nyata pada infeksi berat."
        ],
        "saran_awal": [
            "Buang daun sakit, perbaiki sirkulasi udara kanopi.",
            "Pantau dan terapkan proteksi fungisida bila perlu."
        ]
    },
    "Grape___healthy": {
        "gejala_umum": ["Daun hijau merata tanpa bercak mencolok."],
        "pembeda": ["Tidak ada lesi, tidak ada perubahan warna mencolok."],
        "saran_awal": ["Pertahankan pemeliharaan rutin dan pemantauan berkala."]
    },
    "Apple___Apple_scab": {
        "gejala_umum": [
            "Lesi zaitun/cokelat pada daun, terasa beludru.",
            "Bercak pada buah yang menurunkan kualitas."
        ],
        "pembeda": [
            "Tekstur beludru pada lesi (sporulasi).",
            "Bercak zaitun kehijauan pada awal infeksi."
        ],
        "saran_awal": [
            "Sanitasi daun gugur, atur ventilasi kanopi.",
            "Program fungisida preventif (sesuai rekomendasi)."
        ]
    },
    "Apple___Black_rot": {
        "gejala_umum": [
            "Busuk hitam pada buah; daun ada bercak konsentris.",
            "Canker pada cabang/kayu."
        ],
        "pembeda": [
            "Buah busuk berwarna gelap/menghitam.",
            "Adanya canker pada cabang."
        ],
        "saran_awal": [
            "Pangkas canker, buang buah terinfeksi.",
            "Kelola kebersihan kebun secara ketat."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "gejala_umum": [
            "Bercak jingga/kuning pada daun apel.",
            "Siklus hidup melibatkan juniper/cedar (inang alternatif)."
        ],
        "pembeda": [
            "Bercak oranye terang dengan halo kuning.",
            "Sering terkait keberadaan pohon juniper/cedar di sekitar."
        ],
        "saran_awal": [
            "Jarakkan dari inang alternatif bila memungkinkan.",
            "Terapkan perlindungan fungisida sesuai kondisi lokal."
        ]
    },
    "Apple___healthy": {
        "gejala_umum": ["Daun bersih, hijau merata, tanpa lesi."],
        "pembeda": ["Tidak tampak bercak/nekrosis."],
        "saran_awal": ["Lanjutkan praktik budidaya baik dan pemantauan."]
    },
    "Potato___Early_blight": {
        "gejala_umum": [
            "Bercak cokelat dengan cincin konsentris (target) pada daun.",
            "Sering dimulai pada daun tua, ada halo kuning."
        ],
        "pembeda": [
            "Cincin konsentris jelas (mirip target).",
            "Cenderung menyerang daun yang lebih tua dahulu."
        ],
        "saran_awal": [
            "Buang daun terinfeksi, rotasi tanaman.",
            "Kelola nutrisi & kelembapan; pertimbangkan protektif fungisida."
        ]
    },
    "Potato___Late_blight": {
        "gejala_umum": [
            "Bercak cokelat/abu-abu cepat melebar, tepi tidak beraturan.",
            "Pada kondisi lembap, muncul jamur putih di bawah daun."
        ],
        "pembeda": [
            "Perkembangan sangat cepat saat lembap/dingin.",
            "Miselium putih di bagian bawah daun."
        ],
        "saran_awal": [
            "Segera singkirkan bagian terinfeksi berat.",
            "Program fungisida sesuai rekomendasi; perbaiki drainase."
        ]
    },
    "Potato___healthy": {
        "gejala_umum": ["Daun sehat tanpa bercak."],
        "pembeda": ["Tidak ada lesi atau perubahan warna."],
        "saran_awal": ["Pertahankan monitoring dan praktik budidaya sehat."]
    }
}

# -----------------------------------------------------------------------------
# 3) Utilitas: Preprocessing, Load Model, Prediksi, Penjelasan
# -----------------------------------------------------------------------------
# Transform gambar (harus konsisten dengan training Anda: 224, normalisasi ImageNet)
IMG_SIZE = 224
TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

def _tidy_state_dict_keys(state_dict: dict) -> dict:
    """Bersihkan prefix 'module.' atau 'model.' pada state_dict jika ada."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        elif k.startswith("model."):
            new_sd[k[len("model."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def _find_weights_file(name: str = "best_model_overall.pth") -> Optional[str]:
    """
    Cari file bobot di beberapa lokasi umum:
    - CWD (tempat app dijalankan)
    - folder yang sama dengan app.py
    - subfolder umum: weights/, model/, models/
    - pencarian rekursif dengan rglob
    Return: path absolut (str) jika ketemu, else None.
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = [
        cwd / name,
        here / name,
        cwd / "weights" / name,
        here / "weights" / name,
        cwd / "model" / name,
        here / "model" / name,
        cwd / "models" / name,
        here / "models" / name,
    ]

    # Sertakan hasil rglob (prioritas terakhir)
    try:
        candidates += list(cwd.rglob(name))
    except Exception:
        pass
    if here != cwd:
        try:
            candidates += list(here.rglob(name))
        except Exception:
            pass

    seen = set()
    for p in candidates:
        try:
            p = Path(p).resolve()
        except Exception:
            continue
        if p in seen:
            continue
        seen.add(p)
        if p.exists() and p.is_file():
            return str(p)

    return None

@st.cache_resource(show_spinner=True)
def load_model() -> Tuple[nn.Module, torch.device, bool]:
    """
    Muat arsitektur EfficientNetV2-M (timm) dan coba load bobot lokal jika ada.
    Return:
        model, device, loaded_ok (bool)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Buat model (head disesuaikan untuk 11 kelas)
    model = timm.create_model("efficientnetv2_rw_m", pretrained=False)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, len(CLASS_NAMES))
        )
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        # Head pada beberapa varian adalah Sequential
        # Cari Linear terakhir untuk mengambil in_features
        last_linear = None
        for m in reversed(model.classifier):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError("Tidak menemukan Linear pada head 'classifier'.")
        in_features = last_linear.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, len(CLASS_NAMES))
        )
    else:
        # Fallback: banyak model timm punya attribute 'num_features'
        in_features = getattr(model, "num_features", 1280)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, len(CLASS_NAMES))
        )

    # 🔎 Cari bobot secara cerdas
    weights_path = _find_weights_file("best_model_overall.pth")
    loaded_ok = False

    if weights_path is not None:
        try:
            # Deteksi cepat kasus Git LFS pointer (file sangat kecil dan berisi header khas)
            try:
                with open(weights_path, "rb") as f:
                    head = f.read(2048)
                # Heuristik sederhana pointer LFS
                if (b"git-lfs" in head) or (b"oid sha256:" in head and b"version https://git-lfs.github.com/spec" in head):
                    st.warning(
                        f"File '{weights_path}' tampaknya pointer Git LFS, bukan bobot sebenarnya.\n"
                        "Pastikan LFS ter-fetch (bukan hanya pointer)."
                    )
            except Exception:
                pass

            ckpt = torch.load(weights_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                sd = _tidy_state_dict_keys(ckpt["state_dict"])
            elif isinstance(ckpt, dict) and "model_state" in ckpt:
                sd = _tidy_state_dict_keys(ckpt["model_state"])
            elif isinstance(ckpt, dict):
                sd = _tidy_state_dict_keys(ckpt)
            else:
                sd = ckpt  # bisa jadi langsung state_dict

            model.load_state_dict(sd, strict=False)
            loaded_ok = True
            st.success(f"Bobot dimuat dari: {weights_path}")
        except Exception as e:
            st.warning(
                "⚠️ Gagal memuat bobot. Model berjalan dengan head acak (hasil tidak andal)."
            )
            st.exception(e)
    else:
        # Tampilkan info lokasi yang sudah dicek agar mudah ditelusuri
        st.info(
            "ℹ️ File bobot 'best_model_overall.pth' tidak ditemukan di CWD/script dir/subfolder umum.\n"
            "Letakkan file di folder yang sama dengan app.py atau di subfolder 'weights/'."
        )

    model.to(device)
    model.eval()
    return model, device, loaded_ok

def preprocess(img: Image.Image) -> torch.Tensor:
    """Ubah PIL.Image menjadi tensor siap inferensi."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return TRANSFORM(img).unsqueeze(0)

@torch.inference_mode()
def predict(model: nn.Module, device: torch.device, x: torch.Tensor, topk: int = 3):
    """Lakukan inferensi dan kembalikan (probs_sorted, idx_sorted)."""
    x = x.to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    idx_sorted = np.argsort(probs)[::-1][:topk]
    return probs[idx_sorted], idx_sorted

def build_explanation(label_top: str, prob_top: float, label_next: Optional[str] = None) -> str:
    """
    Bangun penjelasan ringkas berbasis kamus DISEASE_INFO.
    Ini bukan diagnosis, hanya edukasi awal untuk memahami prediksi model.
    """
    info = DISEASE_INFO.get(label_top, None)
    conf_pct = f"{prob_top*100:.1f}%"
    lines = []

    lines.append(f"Prediksi utama: **{label_top}** (keyakinan ~ {conf_pct}).")
    if info:
        if info.get("gejala_umum"):
            lines.append("• Gejala umum yang sering diamati:")
            for g in info["gejala_umum"][:3]:
                lines.append(f"  - {g}")
        if info.get("pembeda"):
            lines.append("• Ciri pembeda yang membantu verifikasi:")
            for p in info["pembeda"][:3]:
                lines.append(f"  - {p}")
        if info.get("saran_awal"):
            lines.append("• Tindakan awal yang bisa dipertimbangkan:")
            for s in info["saran_awal"][:3]:
                lines.append(f"  - {s}")
    else:
        lines.append("• (Belum ada deskripsi terkurasi untuk kelas ini.)")

    if label_next and label_next in DISEASE_INFO:
        lines.append(
            f"Catatan: periksa juga kemungkinan **{label_next}** "
            "jika beberapa gejala mirip atau tumpang tindih."
        )

    lines.append(
        "_Disclaimer: ini bukan pengganti konsultasi agronomi. Verifikasi lapangan tetap diperlukan._"
    )
    return "\n".join(lines)

# -----------------------------------------------------------------------------
# 4) UI: Sidebar (Input Gambar), Info Model, dan Parameter
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Pengaturan & Input")
    uploaded = st.file_uploader(
        "Unggah citra daun (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )
    url_input = st.text_input("atau masukkan URL gambar (opsional)")

    topk = st.slider("Top-K hasil", min_value=1, max_value=5, value=3, step=1)
    run_button = st.button("🔍 Jalankan Diagnosa")

    st.divider()
    st.caption("Kelas target:")
    st.write(", ".join(CLASS_NAMES))

# -----------------------------------------------------------------------------
# 5) Header Halaman
# -----------------------------------------------------------------------------
st.title("🌿 Plant Disease Classifier — CNN + Explainability")
st.markdown(
    "Aplikasi ini mengklasifikasikan penyakit pada daun **Anggur, Apel, dan Kentang** "
    "menggunakan **EfficientNetV2-M**. Setiap prediksi disertai penjelasan ringkas "
    "berbasis pengetahuan gejala umum dan ciri pembeda."
)

# -----------------------------------------------------------------------------
# 6) Muat Model (sekali)
# -----------------------------------------------------------------------------
model, device, loaded_ok = load_model()
status_text = "✅ Bobot terlatih dimuat." if loaded_ok else "⚠️ Bobot tidak tersedia: hasil prediksi tidak andal."
st.toast(status_text)

# -----------------------------------------------------------------------------
# 7) Ambil Gambar Input
# -----------------------------------------------------------------------------
def load_image_from_any() -> Optional[Image.Image]:
    """Ambil gambar dari upload atau URL. Mengembalikan PIL.Image atau None."""
    if uploaded is not None:
        try:
            return Image.open(uploaded)
        except Exception as e:
            st.error("Gagal membaca file yang diunggah.")
            st.exception(e)
            return None
    if url_input:
        try:
            resp = requests.get(url_input, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        except Exception as e:
            st.error("Gagal mengunduh/memuat gambar dari URL.")
            st.exception(e)
            return None
    return None

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("🖼️ Pratinjau Gambar")
    img = load_image_from_any()
    if img is not None:
        st.image(img, use_column_width=True, caption="Gambar input")
    else:
        st.info("Unggah gambar atau isi URL lalu tekan **Jalankan Diagnosa**.")

# -----------------------------------------------------------------------------
# 8) Eksekusi Inferensi & Tampilkan Hasil
# -----------------------------------------------------------------------------
with col_right:
    st.subheader("📊 Hasil Prediksi & Penjelasan")
    if run_button:
        if img is None:
            st.warning("Silakan unggah gambar atau masukkan URL terlebih dahulu.")
        else:
            try:
                x = preprocess(img)
                probs, idxs = predict(model, device, x, topk=topk)

                # Tabel Top-K
                rows = []
                for rank, (p, idx) in enumerate(zip(probs, idxs), start=1):
                    rows.append({
                        "Rank": rank,
                        "Label": CLASS_NAMES[idx],
                        "Prob.": f"{p*100:.2f}%"
                    })
                st.table(rows)

                # Penjelasan ringkas berbasis peta pengetahuan
                label_top = CLASS_NAMES[idxs[0]]
                prob_top = float(probs[0])
                label_next = CLASS_NAMES[idxs[1]] if len(idxs) > 1 else None

                st.markdown(build_explanation(label_top, prob_top, label_next))

                # Indikator reliabilitas bobot
                if not loaded_ok:
                    st.warning(
                        "Model berjalan TANPA bobot terlatih khusus dataset Anda. "
                        "Gunakan hanya untuk demo UI. Untuk hasil akurat, "
                        "letakkan file **best_model_overall.pth** di direktori kerja."
                    )

            except Exception as e:
                st.error("Terjadi kesalahan saat inferensi.")
                st.exception(e)

# -----------------------------------------------------------------------------
# 9) Footnote informasi perangkat
# -----------------------------------------------------------------------------
with st.expander("ℹ️ Informasi Perangkat & Model"):
    st.write(f"Device: **{device.type}**")
    st.write(f"Model: **EfficientNetV2-M** dengan **{len(CLASS_NAMES)}** kelas.")
    st.write("Bobot dimuat:", "✅" if loaded_ok else "❌")
    st.caption(
        "Tip: Tempatkan `best_model_overall.pth` (hasil training Anda) di direktori yang sama "
        "atau di subfolder `weights/` agar prediksi valid. Struktur checkpoint fleksibel: "
        "`state_dict`, `model_state`, atau langsung."
    )

# -----------------------------------------------------------------------------
# 10) Debug Path (opsional; boleh dihapus setelah beres)
# -----------------------------------------------------------------------------
with st.expander("🧪 Debug Path (sementara)"):
    try:
        st.write("CWD:", os.getcwd())
    except Exception as e:
        st.write("CWD: <error>", e)

    try:
        here = Path(__file__).resolve().parent
        st.write("Script dir:", str(here))
    except Exception as e:
        st.write("Script dir: <error>", e)

    try:
        st.write("Isi CWD:", os.listdir("."))
    except Exception as e:
        st.write("Isi CWD: <error>", e)

    try:
        matches = list(Path(".").rglob("best_model_overall.pth"))
        st.write("Rglob matches (CWD):", [str(p) for p in matches])
    except Exception as e:
        st.write("Rglob matches: <error>", e)
