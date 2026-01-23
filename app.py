import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(page_title="Mritunjay's Deepfake Detector", page_icon="🛡️")

# --- LOAD MODEL ---
@st.cache_resource
def load_trained_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    # Using absolute path for your weights
    model.load_state_dict(torch.load(r"D:\deepfake_project\notebooks\deepfake_detector_94acc.pth", map_location='cpu'))
    model.eval()
    return model

model = load_trained_model()

# --- FACE DETECTION FUNCTION ---
def is_human_face(pil_image):
    # Convert PIL image to OpenCV format
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Load OpenCV's built-in face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # minNeighbors=10 makes it stricter (helps reject F1 cars/objects)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return len(faces) > 0

# --- IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UI DESIGN ---
st.title("🛡️ AI Deepfake Detection System")

# PROJECT DISCLAIMER (The Note you wanted)
st.info("""
**📌 Project Note:** This system is optimized for detecting **GAN-generated facial deepfakes**. 
It is designed to analyze human faces only. It may not accurately classify:
- Non-human objects (cars, animals, etc.)
- Diffusion-based AI images (Gemini, Midjourney)
""")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # STEP 1: Run Face Detection Gatekeeper
    with st.spinner("Verifying human face..."):
        face_present = is_human_face(image)
    
    if not face_present:
        st.error("🚨 **No Human Face Detected!** Please upload a clear photo of a person. Objects and animals are not supported.")
    else:
        st.success("✅ Human face verified. Running Deepfake Analysis...")
        
        # STEP 2: Run Inference
        img_t = transform(image).unsqueeze(0)
        
        with torch.inference_mode():
            output = model(img_t)
            prob = torch.nn.functional.softmax(output, dim=1)
            confidence, pred = torch.max(prob, 1)
            
        # STEP 3: Final Result
        label = "REAL" if pred == 1 else "FAKE"
        color = "green" if label == "REAL" else "red"
        
        st.divider()
        st.markdown(f"## Result: :{color}[{label}]")
        st.progress(confidence.item())
        st.write(f"**Confidence Level:** {confidence.item()*100:.2f}%")
