# Eardrum Infection Classification

AI-powered medical image analysis system for classifying eardrum conditions using deep learning.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete ML pipeline
python main.py

# 3. Deploy inference service
python deploy_inference.py

# 4. Access web interface
open http://localhost:8501
```

## Requirements

- **Full Development**: `pip install -r requirements.txt` (PyTorch, MLflow, etc.)
- **Inference API**: `pip install -r requirements-inference.txt` (ONNX Runtime, FastAPI)
- **Web UI**: `pip install -r requirements-web.txt` (Streamlit only)

## What It Does

Classifies eardrum images into 4 medical conditions:
- **Normal**: Healthy eardrum
- **AOM**: Acute Otitis Media (ear infection)
- **Earwax**: Cerumen impaction
- **Chronic**: Chronic Otitis Media

## Project Structure

```
├── main.py                           # Complete ML pipeline entry point
├── export_model.py                   # PyTorch to ONNX model conversion
├── deploy_inference.py               # Inference service deployment
├── run_web.py                        # Web interface runner
├── requirements.txt                  # Main project dependencies
├── requirements-inference.txt        # Inference API dependencies
├── requirements-web.txt              # Web UI dependencies
├── docker-compose.yml                # Multi-service orchestration
├── README.md                         # Project documentation
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── data/                         # Data handling module
│   │   ├── __init__.py
│   │   ├── data_loader.py            # Data loading & preprocessing
│   │   └── dataset_setup.py          # Dataset download & setup
│   ├── training/                     # Model training module
│   │   ├── __init__.py
│   │   ├── model.py                  # EfficientNetV2 Lightning model
│   │   ├── strategies.py             # Fine-tuning strategies
│   │   ├── trainer.py                # Training orchestrator
│   │   └── evaluator.py              # Training evaluation
│   ├── evaluation/                   # Model evaluation module
│   │   ├── __init__.py
│   │   └── evaluator.py              # Final model evaluation
│   └── inference/                    # ONNX inference module
│       ├── __init__.py
│       ├── model_exporter.py         # PyTorch → ONNX conversion
│       ├── onnx_inference.py         # ONNX Runtime engine
│       └── api.py                    # FastAPI service
│
├── web/                              # Web application
│   └── app.py                        # Streamlit web interface
│
├── notebooks/                        # Jupyter notebooks
│   └── ear_infection_classifier.ipynb # Complete ML workflow
│
├── data/                             # Dataset files
│   ├── raw/                          # Original dataset archive
│   │   └── eardrum-dataset-otitis-media.zip
│   ├── eardrum_dataset/              # Extracted dataset (unprocessed)
│   │   ├── Aom/                      # Acute Otitis Media (119 images)
│   │   ├── Chornic/                  # Chronic Otitis Media (63 images)
│   │   ├── Earwax/                   # Cerumen impaction (140 images)
│   │   ├── Normal/                   # Healthy eardrums (534 images)
│   │   └── [excluded classes]/       # Low-sample classes
│   ├── eardrum_split/                # Processed dataset splits
│   │   ├── train/                    # Training images (70%)
│   │   ├── val/                      # Validation images (15%)
│   │   └── test/                     # Test images (15%)
│   └── processed/                    # Dataset metadata
│       ├── class_mapping.csv         # Class name mappings
│       ├── dataset_info.csv          # Dataset statistics
│       ├── train_split.csv           # Training set file list
│       ├── val_split.csv             # Validation set file list
│       └── test_split.csv            # Test set file list
│
├── checkpoints/                      # Model checkpoints
│   └── best_checkpoints_*.json       # Best model checkpoint files
│
├── models/                           # Exported models
│   └── onnx/                         # ONNX models for inference
│       ├── eardrum_classifier_full.onnx
│       ├── eardrum_classifier_full_metadata.json
│       └── model_mapping.json
│
├── visualizations/                   # Generated plots and charts
│   └── confusion_matrices/           # Confusion matrix images
│       ├── confusion_matrix_freeze_backbone.png
│       ├── confusion_matrix_full.png
│       ├── confusion_matrix_last1+head.png
│       ├── confusion_matrix_last2+head.png
│       ├── confusion_matrix_last3+head.png
│       └── confusion_matrix_last4+head.png
│
├── mlruns/                           # MLflow experiment tracking
│   ├── */                            # Experiment runs (UUID folders)
│   │   ├── */                        # Individual run folders
│   │   ├── meta.yaml                 # Experiment metadata
│   │   └── models/                   # Registered models
│   └── models/                       # Model registry
│
├── logs/                             # Application logs
│
└── docker/                           # Containerization
    ├── Dockerfile                    # Main training container
    ├── Dockerfile.inference          # Inference API container
    ├── Dockerfile.web                # Web UI container
    └── README.md                     # Docker documentation
```

### Key Directories

- **`src/`**: Modular source code organized by functionality
- **`data/`**: Complete dataset pipeline from raw to processed
- **`web/`**: Streamlit web interface for testing
- **`notebooks/`**: Jupyter notebooks for analysis and experimentation
- **`checkpoints/`**: Model checkpoints and training artifacts
- **`models/`**: Exported ONNX models for production inference
- **`visualizations/`**: Generated plots, charts, and confusion matrices
- **`mlruns/`**: MLflow experiment tracking and model registry
- **`docker/`**: Containerization files for deployment

## Usage

### Training
```bash
python main.py
```

### Web Interface
```bash
# Start all services
docker-compose up -d

# Access web interface
open http://localhost:8501
```

### API
```bash
# Health check
curl http://localhost:8000/health

# Predict from image
curl -X POST 'http://localhost:8000/predict' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@eardrum_image.jpg'
```

## Technical Details

- **Model**: EfficientNetV2-S (PyTorch Lightning)
- **Inference**: ONNX Runtime (FastAPI)
- **Web UI**: Streamlit
- **Deployment**: Docker Compose
- **Tracking**: MLflow

## Dataset

- **598 TIFF images** (224×224 pixels)
- **4 main classes** (filtered from 9 original)
- **Train/Val/Test split**: 70%/15%/15%
- **Medical-grade quality** with proper preprocessing

## Requirements

- Python 3.9+
- PyTorch
- ONNX Runtime
- Streamlit
- Docker (optional)

## Medical Disclaimer

This tool is for research and educational purposes only. It should not be used for actual medical diagnosis. Always consult with qualified healthcare professionals.

## Support

- **API Docs**: http://localhost:8000/docs
- **Issues**: Create GitHub issue
- **Logs**: `docker-compose logs -f`
