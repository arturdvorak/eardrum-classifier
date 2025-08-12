# Eardrum Classification with EfficientNetV2 + PyTorch Lightning

This project builds a robust classifier for tympanic membrane (eardrum) conditions using medical images. It leverages a pre-trained EfficientNetV2-S model from `timm`, fine-tunes it with PyTorch Lightning, and evaluates multiple transfer learning strategies.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "Image Recognision"

# Install the package in development mode
pip install -e .
```

### Usage

#### 1. **Pipeline Script** (Recommended for full workflow)
```bash
# Run the complete pipeline
python scripts/run_pipeline.py --step all

# Run specific steps
python scripts/run_pipeline.py --step data      # Data setup only
python scripts/run_pipeline.py --step train     # Training only
python scripts/run_pipeline.py --step evaluate  # Evaluation only

# Get help
python scripts/run_pipeline.py --help
```

#### 2. **CLI Tools** (Individual operations)
```bash
# Training
eardrum-train

# Evaluation
eardrum-evaluate

# Prediction
eardrum-predict --image path/to/image.jpg --output predictions.json

# Get help for any tool
eardrum-train --help
eardrum-evaluate --help
eardrum-predict --help
```

#### 3. **Python Import**
```python
from eardrum_classifier import EfficientNetV2Lightning

# Create model
model = EfficientNetV2Lightning(
    num_classes=6,  # Aom, Chronic, Earwax, Normal, Otitis Externa, Tympanosclerosis
    lr=1e-4
)
```

---

## 📁 Project Structure

```
Image Recognision/
├── config/                 # Configuration files
├── data/                   # Dataset directories
│   ├── external/          # Raw downloaded data
│   ├── processed/         # Processed and split data
│   └── raw/              # Original dataset
├── docker/                # Docker configuration
├── docs/                  # Documentation
├── logs/                  # Training logs and MLflow tracking
├── models/                # Saved models and checkpoints
├── notebooks/             # Jupyter notebooks
├── requirements/          # Dependencies
├── scripts/               # Executable scripts
│   ├── data_setup.py     # Data preparation
│   ├── train.py          # Model training
│   ├── evaluate.py        # Model evaluation
│   ├── predict.py         # Inference
│   └── run_pipeline.py    # Complete pipeline
├── src/                   # Source code package
│   └── eardrum_classifier/
│       ├── data/          # Data loading and transforms
│       ├── models/        # Model definitions
│       ├── training/      # Training utilities
│       └── utils/         # Utility functions
├── tests/                 # Test files
├── web/                   # Web application
└── setup.py              # Package configuration
```

---

## 🎯 Features

### ✅ **Working Components**
- **CLI Tools**: `eardrum-train`, `eardrum-evaluate`, `eardrum-predict`
- **Pipeline Script**: Complete workflow orchestration
- **Python Package**: Importable `eardrum_classifier` module
- **Model Architecture**: EfficientNetV2 with PyTorch Lightning
- **Training Strategies**: Multiple fine-tuning approaches
- **MLflow Integration**: Experiment tracking and model management

### 🔄 **Training Strategies**
- `freeze_backbone`: Only head is trainable
- `last1+head`, `last2+head`, `last3+head`, `last4+head`: Gradually unfreeze blocks
- `full`: All layers trainable

### 📊 **Evaluation Metrics**
- **Primary**: Macro F1 Score (robust to class imbalance)
- **Secondary**: Accuracy, Precision, Recall
- **Visualization**: Confusion matrices, performance plots

---

## Dataset

- **Source**: [Kaggle – Eardrum Dataset (Otitis Media)](https://www.kaggle.com/datasets/erdalbasaran/eardrum-dataset-otitis-media)  
- **Total images**: 956  
- **Classes Used**:
  - Normal
  - Acute Otitis Media (AOM)
  - Chronic Otitis Media
  - Otitis Externa
  - Earwax
  - Tympanosclerosis
- **Excluded classes (due to very low samples)**:
  - Foreign
  - PseduoMembran
  - Earventulation

> **Cycle 2 classes only**:
> - Aom  
> - Chronic  
> - Earwax  
> - Normal

---

## 🔧 Technical Details

### Model Architecture
- **Base Model**: EfficientNetV2-S from `timm`
- **Input Size**: 224×224 pixels
- **Normalization**: ImageNet mean/std
- **Optimizer**: AdamW with ReduceLROnPlateau
- **Loss**: CrossEntropyLoss

### Data Pipeline
- **Augmentation**: Horizontal flip, rotation, normalization
- **Split Ratio**: 70% train / 15% validation / 15% test
- **Batch Size**: Configurable via configuration files

### Training Features
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best models
- **Learning Rate Scheduling**: Adaptive LR adjustment
- **MLflow Tracking**: Experiment logging and comparison

---

## 📈 Usage Examples

### Training a Single Strategy
```bash
# Train with freeze_backbone strategy
eardrum-train

# Check training progress
tail -f logs/pipeline.log
```

### Evaluating Models
```bash
# Evaluate all trained strategies
eardrum-evaluate

# Results will be saved to logs/evaluation_results.json
```

### Making Predictions
```bash
# Single image prediction
eardrum-predict --image data/test/sample.jpg --output prediction.json

# Batch prediction
eardrum-predict --images data/test/*.jpg --output batch_predictions.json
```

### Running Complete Pipeline
```bash
# Run everything
python scripts/run_pipeline.py --step all --verbose

# Run with custom config
python scripts/run_pipeline.py --step all --config config/custom_config.yaml
```

---

## 📚 Dependencies

### Core ML Libraries
- PyTorch ≥ 2.0
- PyTorch Lightning ≥ 2.0
- torchvision
- timm
- torchmetrics

### Data Processing
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Experiment Tracking
- MLflow
- logging

### Development
- pytest
- black
- flake8
- mypy

---

*Last updated: August 2025*