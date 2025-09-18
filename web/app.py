"""
Eardrum Classification Web Interface

A Streamlit web application for testing the eardrum infection classification model.
Provides an intuitive interface for uploading images and viewing predictions.
"""

import streamlit as st
import requests
import json
import io
from PIL import Image
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Eardrum Infection Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .class-normal {
        color: #28a745;
        font-weight: bold;
    }
    .class-abnormal {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://eardrum-inference:8000")

def check_api_health():
    """Check if the inference API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

def predict_image(image_file):
    """Send image to the inference API for prediction."""
    try:
        files = {"image": image_file}
        response = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection Error: {str(e)}"

def get_model_info():
    """Get model information from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

def get_confidence_class(confidence):
    """Get CSS class for confidence level."""
    if confidence >= 0.8:
        return "confidence-high"
    elif confidence >= 0.6:
        return "confidence-medium"
    else:
        return "confidence-low"

def get_class_style(prediction):
    """Get CSS class for prediction type."""
    if prediction.lower() == "normal":
        return "class-normal"
    else:
        return "class-abnormal"

def create_probability_chart(probabilities):
    """Create a bar chart for class probabilities."""
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Color mapping
    colors = ['#28a745' if cls.lower() == 'normal' else '#dc3545' for cls in classes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Ear Condition",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">Eardrum Infection Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Medical Image Analysis</p>', unsafe_allow_html=True)
    
    # Check API health
    with st.spinner("Checking API connection..."):
        api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("**API Connection Failed**")
        st.error("Please ensure the inference service is running on port 8000")
        st.code("docker-compose up -d eardrum-inference", language="bash")
        return
    
    st.success("**API Connected Successfully**")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Information")
        
        model_info = get_model_info()
        if model_info:
            st.write(f"**Model Path:** {model_info['model_path'].split('/')[-1]}")
            st.write(f"**Input Shape:** {model_info['input_shape']}")
            st.write(f"**Classes:** {', '.join(model_info['class_names'])}")
            st.write(f"**Providers:** {', '.join(model_info['providers'])}")
        
        st.header("About This Tool")
        st.write("""
        This AI-powered tool helps classify eardrum conditions from medical images.
        
        **Supported Conditions:**
        - **Normal**: Healthy eardrum
        - **AOM**: Acute Otitis Media (ear infection)
        - **Earwax**: Cerumen impaction
        - **Chronic**: Chronic Otitis Media
        
        **How to Use:**
        1. Upload an eardrum image
        2. Click 'Analyze Image'
        3. View the AI prediction and confidence
        """)
        
        st.header("Medical Disclaimer")
        st.warning("""
        This tool is for research and educational purposes only. 
        It should not be used for actual medical diagnosis. 
        Always consult with qualified healthcare professionals.
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an eardrum image",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            help="Upload a clear image of an eardrum for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
            st.write(f"**File Size:** {uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.header("Analysis Results")
        
        if uploaded_file is not None:
            # Analyze button
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Get prediction
                    success, result = predict_image(uploaded_file)
                
                if success:
                    # Display results
                    prediction = result['prediction']
                    confidence = result['confidence']
                    probabilities = result['probabilities']
                    processing_time = result.get('processing_time_ms', 0)
                    
                    # Main prediction
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Prediction: <span class="{get_class_style(prediction)}">{prediction}</span></h3>
                        <h4>Confidence: <span class="{get_confidence_class(confidence)}">{confidence:.1%}</span></h4>
                        <p>Processing Time: {processing_time:.1f}ms</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability chart
                    st.plotly_chart(create_probability_chart(probabilities), use_container_width=True)
                    
                    # Detailed probabilities
                    st.subheader("Detailed Probabilities")
                    for class_name, prob in probabilities.items():
                        progress = prob
                        st.write(f"**{class_name}:** {prob:.1%}")
                        st.progress(progress)
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    if prediction.lower() == "normal":
                        st.success("The AI detected a normal, healthy eardrum appearance.")
                    else:
                        st.warning(f"The AI detected potential abnormalities consistent with {prediction}.")
                        st.info("Please consult with a healthcare professional for proper diagnosis.")
                
                else:
                    st.error(f"**Analysis Failed**")
                    st.error(result)
        else:
            st.info("Please upload an image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Eardrum Infection Classifier | Powered by ONNX Runtime | Built with Streamlit</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()