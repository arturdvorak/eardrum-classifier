# app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from datetime import datetime
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from model import EfficientNetV2Lightning  # Use your actual model class

# ------------ Config ------------
CHECKPOINT_PATH = "/Users/arturdvorak/Desktop/ML course/Notebooks/Image Recognision/mlruns/834690443400753513/4aca2af8a020429183255415bc380ee0/checkpoints/last2+head-best-f1-epoch=03-val_f1=0.4978.ckpt"
NUM_CLASSES = 6
CLASS_NAMES = [
    "Aom",
    "Chornic",
    "Earwax",
    "Normal",
    "OtitExterna",
    "tympanoskleros"
]
IMAGE_SIZE = 224
WEBHOOK_URL = ""  # Optional webhook URL
# --------------------------------


@st.cache_resource
def load_model():
    model = EfficientNetV2Lightning.load_from_checkpoint(
        CHECKPOINT_PATH,
        num_classes=NUM_CLASSES
    )
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


def send_webhook(label: str, prob: float):
    payload = {
        "timestamp": datetime.now().isoformat(),
        "label": label,
        "confidence": round(prob * 100, 2)
    }
    try:
        response = requests.post(WEBHOOK_URL, json=payload)
        if response.status_code == 200:
            st.success("Webhook sent successfully.")
        else:
            st.warning(f"Webhook failed: {response.status_code}")
    except Exception as e:
        st.error(f"Error sending webhook: {e}")


# ------------- UI -------------
st.title("Eardrum Classifier")
st.write("Upload an image of the tympanic membrane to classify its condition.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

use_webhook = st.checkbox("Send result to Microsoft Flow webhook")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        model = load_model()
        input_tensor = preprocess_image(image)

    with torch.no_grad():
        device = next(model.parameters()).device  # Get model device (e.g., mps or cpu)
        input_tensor = input_tensor.to(device)    # Move input to same device
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = CLASS_NAMES[pred.item()]
        confidence = conf.item()

    st.success(f"Prediction: {label} ({confidence * 100:.2f}%)")

    if use_webhook and WEBHOOK_URL:
        send_webhook(label, confidence)
# ------------------------------