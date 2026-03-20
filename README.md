# 🛡️ Deepfake Face Detector

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Accuracy](https://img.shields.io/badge/Accuracy-94.05%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Model](https://img.shields.io/badge/Model-ResNet--50-orange)

A state-of-the-art Deep Learning pipeline for **real-time binary classification of facial forgeries**.  
Developed as a **core portfolio project** for the **Associate of Science in Artificial Intelligence** at **Saras AI Institute**, with a focus on detecting **GAN-generated facial artifacts** with high precision.

---

## 📊 Performance at a Glance

| Metric | Score |
|---|---|
| ✅ Final Test Accuracy | **94.05%** |
| ✅ Validation Accuracy | **93.12%** (after 5 epochs) |
| ✅ Precision (Fake) | **0.95** |
| ✅ Recall (Fake) | **0.93** |
| ✅ ROC-AUC | **~0.98** |
| ✅ Inference Latency | **< 12 ms** per image |
| 📦 Dataset | **140,000** high-resolution images (Kaggle) |

---

## 🧠 Model Architecture & Methodology

The project uses **Transfer Learning** with a pre-trained **ResNet-50** backbone, fine-tuned for **digital image forensics**.

### 🔧 Custom Classification Head

```
ResNet-50 Backbone (ImageNet pretrained)
        ↓
Global Average Pooling
        ↓
Fully Connected Layer (2048 → 512, ReLU)
        ↓
Dropout (0.3) — prevents overfitting on GAN artifacts
        ↓
Output Layer — Softmax (REAL vs FAKE)
```

### Why ResNet-50?
- Deep residual connections prevent vanishing gradients on a 140K image dataset
- ImageNet pretraining gives strong low-level feature extraction out of the box
- Lightweight enough for < 12ms inference on consumer hardware

---

## 💻 Hardware & Training Specifications

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 (8 GB VRAM) |
| Storage | Gen 5 SSD (high-speed data loading) |
| Framework | PyTorch |
| Optimization | Mixed Precision Training (`torch.cuda.amp`) |
| Training Time | ~30 minutes for 5 epochs |
| Dataset Size | 140,000 images (real + GAN-generated) |

---

## 📈 Evaluation Results

The model was evaluated using metrics beyond simple accuracy to ensure reliability in real-world forensic scenarios:

- **Precision (Fake): 0.95** — Minimizes false alarms on real people
- **Recall (Fake): 0.93** — Catches most digital forgeries
- **ROC-AUC: ~0.98** — Near-perfect class separation between real and fake

---

## 📦 Model Weights

The trained model weights (`deepfake_detector_model.pth`) are **~93.99 MB** — exceeding GitHub's standard upload limit.

Model weights are hosted in the **Releases** section of this repository.

### 🔽 How to Download

1. Go to the [**Releases Page**](https://github.com/mritunjaySharma05/Deepfake-Face-Detector/releases/tag/v1.0.0)
2. Download `deepfake_detector_model.pth`
3. Place the file in the **root directory** of the project
4. Run the app — it will load automatically

---

## 🛠️ Project Structure

```
Deepfake-Face-Detector/
│
├── app.py                        # Streamlit web interface for live detection
├── deepfake.ipynb                # Training, evaluation & research notebook
├── requirements.txt              # Environment dependencies
├── .gitignore                    # Prevents checkpoint/cache uploads
└── deepfake_detector_model.pth   # Download from Releases (not in repo)
```

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/mritunjaySharma05/Deepfake-Face-Detector.git
cd Deepfake-Face-Detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download model weights
Download `deepfake_detector_model.pth` from [Releases](https://github.com/mritunjaySharma05/Deepfake-Face-Detector/releases/tag/v1.0.0) and place it in the root directory.

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Open in browser
```
http://localhost:8501
```

Upload any face image or video — the model will instantly classify it as **REAL** or **FAKE**.

---

## 🧪 How It Works

```
Input Image / Video Frame
        ↓
Preprocessing (Resize → Normalize → Tensor)
        ↓
ResNet-50 Feature Extraction
        ↓
Custom Classification Head
        ↓
Probability Score → REAL or FAKE
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | ResNet-50 (Transfer Learning) |
| Framework | PyTorch |
| Training | Mixed Precision (`torch.cuda.amp`) |
| Interface | Streamlit |
| Hardware | NVIDIA RTX 4060 (8GB VRAM) |
| Language | Python 3.10+ |

---

## 👤 Author

**Mritunjay Sharma** — AI & ML Engineer  
Associate of Science in AI | Saras AI Institute  

[GitHub](https://github.com/mritunjaySharma05) · [LinkedIn](https://linkedin.com/in/mritunjay-sharma05)

---

> ⭐ Star this repo if you found it useful!
