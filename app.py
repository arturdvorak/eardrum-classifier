# app.py

# ========== Standard & Core ==========
import os
import torch
import requests
from datetime import datetime
from PIL import Image

# ========== Image Processing ==========
import torchvision.transforms as transforms

# ========== Streamlit ==========
import streamlit as st

# ========== Your Model ==========
from model import EfficientNetV2Lightning

# ------------ Config ------------
CHECKPOINT_PATH = "best_model.ckpt"
DRIVE_FILE_ID = "1YE2TYrruX4kVKtnwwXmXz7VKXCYjrVlf"
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


def download_file_from_google_drive(file_id, destination):
    """Downloads large files from Google Drive with token support (no auth)."""
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


@st.cache_resource
def load_model():
    """Download checkpoint (if needed) and load model."""
    if not os.path.exists(CHECKPOINT_PATH):
        st.info("Downloading model checkpoint from public Google Drive...")
        download_file_from_google_drive(DRIVE_FILE_ID, CHECKPOINT_PATH)
        st.success("Model checkpoint downloaded successfully.")

    model = EfficientNetV2Lightning.load_from_checkpoint(
        CHECKPOINT_PATH,
        num_classes=NUM_CLASSES
    )
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    """Resize, normalize, and convert image to tensor."""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def send_webhook(label: str, prob: float):
    """Send prediction result to webhook if enabled."""
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


# ========== Streamlit UI ==========
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
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)
        label = CLASS_NAMES[pred.item()]
        confidence = conf.item()

    st.success(f"Prediction: {label} ({confidence * 100:.2f}%)")

    if use_webhook and WEBHOOK_URL:
        send_webhook(label, confidence)