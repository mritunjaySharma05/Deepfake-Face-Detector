# 🛡️ Deepfake Face Detector

A state-of-the-art Deep Learning pipeline for **real-time binary classification of facial forgeries**.  
This project was developed as a **core portfolio project** for the **Associate of Science in Artificial Intelligence** at **Saras AI Institute**, with a focus on detecting **GAN-generated facial artifacts** with high precision.

---

## 📊 Performance at a Glance

- **Final Test Accuracy:** 94.05%
- **Validation Accuracy:** 93.12% (after 5 epochs)
- **Dataset:** 140,000 high-resolution real & fake face images (Kaggle)
- **Inference Latency:** < 12 ms per image on local hardware

---

## 🧠 Model Architecture & Methodology

The project leverages **Transfer Learning** using a pre-trained **ResNet-50** backbone.  
Features learned from **ImageNet** were fine-tuned for the task of **digital image forensics**.

### 🔧 Custom Classification Head

To adapt ResNet-50 for binary deepfake detection, the following head was implemented:

- **Global Average Pooling** – reduces spatial dimensions while preserving semantic features
- **Fully Connected Layer** – 2048 → 512 neurons (ReLU activation)
- **Regularization** – Dropout (0.3) to prevent overfitting on GAN artifacts
- **Output Layer** – Softmax for binary classification (**REAL vs FAKE**)

---
## 📦 Model Weights & Releases

The trained model weights (`deepfake_detector_model.pth`) are approximately **93.99 MB**, which exceeds GitHub’s standard web upload limit.

To keep the repository clean and lightweight, the model weights are hosted under the **Releases** section of this repository.

### 🔽 How to Use the Trained Model

1. Navigate to the [Releases Page](https://github.com/mritunjaySharma05/Deepfake-Face-Detector/releases/tag/v1.0.0)of this repository.
2. Download `deepfake_detector_model.pth`.
3. Place the file in the **root directory** of the project.
4. Run the application as usual.

Once placed correctly, the Streamlit app will automatically load the model for inference.

---

## 💻 Hardware & Engineering Specifications

Training a dataset of 140k images requires significant compute resources.  
The model was trained on a **local workstation** for maximum throughput and control.

- **GPU:** NVIDIA GeForce RTX 4060 (8 GB VRAM)
- **Storage:** Gen 5 SSD (high-speed data loading)
- **Framework:** PyTorch
- **Optimization:** Mixed Precision Training (`torch.cuda.amp`)
- **Training Time:** ~30 minutes for 5 epochs

---

## 🛠️ Project Structure

```text
├── app.py                       # Streamlit web interface for live testing
├── deepfake.ipynb               # Research, training, and evaluation notebook
├── requirements.txt             # Environment dependencies
├── .gitignore                   # Prevents junk/checkpoint uploads
└── deepfake_detector_model.pth  # To be downloaded from Releases
```
## 🚀 Installation & Setup

1. Clone the repository
```bash
git clone [https://github.com/mritunjay-sharma/Deepfake-Face-Detector.git](https://github.com/mritunjay-sharma/Deepfake-Face-Detector.git)
cd Deepfake-Face-Detector
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the Application:
```
streamlit run app.py
```
## 📈 Evaluation Results

The model's performance was rigorously evaluated using metrics beyond simple 
- accuracy:Precision (Fake): 0.95 — Minimizing "False Alarms" on real people.
- Recall (Fake): 0.93 — Ensuring most digital forgeries are caught.
- ROC-AUC: ~0.98 — Demonstrating near-perfect class separation.
## 👤 Author 
**Mritunjay Sharma** | Associate of Science in AI | Saras AI Institute | 
[LinkedIn](www.linkedin.com/in/mritunjay-sharma05)


