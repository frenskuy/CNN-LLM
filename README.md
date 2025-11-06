# CNN-LLM: Penjelasan Otomatis Diagnosis Penyakit Daun 🌿🤖

[![Open in Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-brightgreen)](https://llm-explain.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-success.svg)](#contributing)

> **TL;DR**: Proyek ini menggabungkan **CNN** untuk klasifikasi penyakit daun dan **LLM** untuk menghasilkan **penjelasan tekstual** yang informatif berdasarkan label hasil klasifikasi. Live demo tersedia di **Streamlit**: **https://llm-explain.streamlit.app/**

---

## ✨ Fitur Utama

- **Klasifikasi Citra Daun (CNN)** — Fine-tuning EfficientNetV2 (via `timm`), terlatih pada subset PlantVillage (Apple/Grape/Potato).
- **Penjelasan Otomatis (LLM)** — Default `google/flan-t5-large` + templat prompt + **`disease_info.json`** (gejala, pembeda, saran awal).
- **Akurasi + Keterjelasan** — Prediksi disertai confidence score dan uraian alasan mengapa gambar masuk ke label tersebut.
- **Aplikasi Web Interaktif** — Antarmuka **Streamlit** untuk upload gambar ➜ diagnosis ➜ penjelasan.

---

## 🚀 Demo

- **Live App**: **https://llm-explain.streamlit.app/**
- **Cara pakai**: unggah foto daun ➜ sistem memprediksi label penyakit ➜ LLM menulis penjelasan + saran awal.

---

## 🧠 Arsitektur (Ringkas)

```
[Input Gambar]
      │
      ▼
  Preprocess (Resize/Normalize)
      │
      ▼
 EfficientNetV2  ──►  label + confidence
      │
      │    ┌─────────────────────────────────────────────────┐
      └──► │  disease_info[label] + Template Prompt + LLM    │
           └─────────────────────────────────────────────────┘
                              │
                              ▼
                   Penjelasan tekstual (LLM)
```

---

## ⚙️ Instalasi Cepat (Lokal)

> **Catatan bobot model**: file `*.pth` dilacak dengan **Git LFS** (~200MB+). Pastikan LFS terpasang agar bobot otomatis terunduh.

```bash
git clone https://github.com/frenskuy/CNN-LLM.git
cd CNN-LLM

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

git lfs install
git lfs pull

streamlit run app.py
```

**Environment (opsional)**  
- `LLM_MODEL_NAME` (default: `google/flan-t5-large`)  
- `DEVICE` = `cuda`/`cpu` (otomatis terdeteksi jika tidak diset)

---

## 📈 Hasil (Classification Report)

> Ringkasan performa model pada set uji, ditulis dalam format seperti keluaran `sklearn.metrics.classification_report`.

```
Classification Report:

                                          precision    recall  f1-score   support

               Apple___Apple_scab            1.0000    1.0000    1.0000       116
                 Apple___Black_rot           1.0000    0.9917    0.9959       121
        Apple___Cedar_apple_rust             1.0000    1.0000    1.0000        56
                 Apple___healthy             0.9970    1.0000    0.9985       327
                 Grape___Black_rot           0.9872    0.9957    0.9914       232
        Grape___Esca_(Black_Measles)         0.9962    0.9887    0.9925       266
Grape___Leaf_blight_(Isariopsis_Leaf_Spot)   1.0000    1.0000    1.0000       228
                 Grape___healthy             1.0000    1.0000    1.0000        80
             Potato___Early_blight           1.0000    1.0000    1.0000       210
              Potato___Late_blight           0.9951    1.0000    0.9975       202
                Potato___healthy             1.0000    0.9744    0.9870        39

                      accuracy                                   0.9968      1877
                     macro avg               0.9978    0.9955    0.9966      1877
                  weighted avg               0.9968    0.9968    0.9968      1877
```

## 🧪 Reproduksi Evaluasi (contoh)

```python
# contoh: scripts/eval.py
import torch, json
from sklearn.metrics import classification_report
from src.inference import load_cnn
from src.transforms import get_eval_transform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

device = "cuda" if torch.cuda.is_available() else "cpu"
cnn = load_cnn("best_model_overall.pth", device=device)

ds = ImageFolder("data/test", transform=get_eval_transform(size=224))
dl = DataLoader(ds, batch_size=64, num_workers=4)

y_true, y_pred = [], []
with torch.inference_mode():
    for x, y in dl:
        x = x.to(device)
        logits = cnn(x)
        y_hat = logits.argmax(1).cpu().tolist()
        y_pred += y_hat
        y_true += y.cpu().tolist()

target_names = ds.classes
print(\"Classification Report:\\n\")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
```

---

## 🗃️ Struktur Proyek (ringkas)

```
CNN-LLM/
├─ app.py
├─ disease_info.json
├─ best_model_overall.pth    # Git LFS
├─ requirements.txt
├─ src/
│  ├─ models.py
│  ├─ inference.py
│  ├─ transforms.py
│  └─ prompt.py
└─ README.md
```

---

## 🤝 Kontribusi

PR & issue terbuka lebar. Gunakan branch feature (`feat/...`) dan sertakan ringkasan perubahan.

---

## 📄 Lisensi

MIT — bebas digunakan untuk riset maupun pengembangan lebih lanjut.

---

## 🙏 Apresiasi

- PlantVillage dataset
- PyTorch, timm, Hugging Face Transformers, Streamlit
- Komunitas open-source
