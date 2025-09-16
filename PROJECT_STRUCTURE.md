# Project Structure

This is the current project structure for the Eardrum Infection Classification project - a medical AI system for automatically diagnosing ear conditions from eardrum images.

## Current Structure

```
├── main.py                                    # Main entry point - complete ML pipeline
├── requirements.txt                           # Python dependencies
├── PROJECT_STRUCTURE.md                      # This documentation file
├── .gitignore                                # Git ignore rules
├── .gitattributes                            # Git attributes
├── checkpoints/                              # Model checkpoints and training artifacts
│   └── best_checkpoints_*.json              # Best model checkpoint files
├── data/                                     # All data organized here
│   ├── raw/                                  # Original, unprocessed data
│   │   └── *.zip                           # Original dataset archive
│   ├── eardrum_dataset/                      # Extracted dataset (unprocessed)
│   │   ├── Aom/                             # Acute Otitis Media (119 images)
│   │   ├── Chornic/                         # Chronic Otitis Media (63 images)
│   │   ├── Earventulation/                  # Ear ventilation (16 images)
│   │   ├── Earwax/                          # Cerumen impaction (140 images)
│   │   ├── Foreign/                         # Foreign objects (3 images)
│   │   ├── Normal/                          # Healthy eardrums (534 images)
│   │   ├── OtitExterna/                     # External otitis (41 images)
│   │   ├── PseduoMembran/                   # Pseudomembrane (11 images)
│   │   └── tympanoskleros/                  # Tympanosclerosis (28 images)
│   ├── eardrum_split/                        # Processed dataset splits
│   │   ├── train/                           # Training images (70%)
│   │   │   ├── Aom/                         # 83 images
│   │   │   ├── Chornic/                     # 44 images
│   │   │   ├── Earwax/                      # 98 images
│   │   │   └── Normal/                      # 373 images
│   │   ├── val/                             # Validation images (15%)
│   │   │   ├── Aom/                         # 17 images
│   │   │   ├── Chornic/                     # 9 images
│   │   │   ├── Earwax/                      # 21 images
│   │   │   └── Normal/                      # 80 images
│   │   └── test/                            # Test images (15%)
│   │       ├── Aom/                         # 19 images
│   │       ├── Chornic/                     # 10 images
│   │       ├── Earwax/                      # 21 images
│   │       └── Normal/                      # 81 images
│   └── processed/                           # Dataset metadata and file lists
│       ├── class_mapping.csv                # Class name mappings
│       ├── dataset_info.csv                 # Dataset statistics
│       ├── train_split.csv                  # Training set file list
│       ├── val_split.csv                    # Validation set file list
│       └── test_split.csv                   # Test set file list
├── visualizations/                          # Generated plots and visualizations
│   └── confusion_matrices/                  # Confusion matrix plots
│       └── confusion_matrix_*.png           # Confusion matrix images
├── mlruns/                                  # MLflow experiment tracking
│   ├── */                                   # Experiment runs (UUID folders)
│   │   ├── */                              # Individual run folders
│   │   ├── meta.yaml                       # Experiment metadata
│   │   └── models/                         # Registered models
│   └── models/                             # Model registry
├── notebooks/                               # Jupyter notebooks
│   └── *.ipynb                             # Analysis notebooks
├── src/                                     # Source code package
│   ├── __init__.py                         # Package initialization
│   ├── data/                               # Data handling module
│   │   ├── __init__.py
│   │   ├── data_loader.py                  # Data loading & preprocessing
│   │   └── dataset_setup.py                # Dataset download & setup
│   ├── training/                           # Model training module
│   │   ├── __init__.py
│   │   ├── evaluator.py                    # Training evaluation
│   │   ├── model.py                        # EfficientNetV2 Lightning model
│   │   ├── strategies.py                   # Fine-tuning strategies
│   │   └── trainer.py                      # Training orchestrator
│   └── evaluation/                         # Model evaluation module
│       ├── __init__.py
│       └── evaluator.py                    # Final model evaluation
├── web/                                     # Web application
│   └── app.py                              # Streamlit web interface
└── docker/                                  # Containerization
    └── Dockerfile                          # Docker configuration
```

## Key Features

### Medical AI Focus
- **Domain-specific**: Designed for ear infection diagnosis
- **4 Main Classes**: Aom, Earwax, Normal, Chornic (Chronic)
- **Class Imbalance Handling**: Uses F1-score as primary metric
- **Clinical Relevance**: Real medical conditions being classified

### Production Ready
- **Complete Pipeline**: `python main.py` runs everything
- **Modular Design**: Clean separation of data, training, evaluation
- **Error Handling**: Robust error management throughout
- **Web Interface**: Streamlit app for real-time predictions
- **Docker Support**: Containerized deployment ready

### Advanced Experimentation
- **6 Fine-tuning Strategies**: From freeze_backbone to full training
- **MLflow Integration**: Comprehensive experiment tracking
- **Multiple Checkpoints**: Historical model versions preserved
- **Best Model Selection**: Automatic based on validation F1-score

### Comprehensive Dataset
- **Original Dataset**: All 9 classes preserved in `data/eardrum_dataset/`
- **Processed Splits**: Clean train/val/test in `data/eardrum_split/`
- **Metadata Tracking**: CSV files for dataset statistics
- **Organized Structure**: All data properly organized in `data/` folder

## Dataset Details

### Medical Classes (4 Main Categories)
- **Aom (Acute Otitis Media)**: Ear infection with fluid buildup behind eardrum
- **Earwax**: Cerumen impaction blocking the ear canal
- **Normal**: Healthy eardrum with no visible abnormalities
- **Chornic (Chronic Otitis Media)**: Long-term ear infection with persistent symptoms

### Excluded Classes (Low Sample Count)
- **Earventulation**: Ear ventilation tubes (16 images)
- **tympanoskleros**: Tympanosclerosis (28 images)
- **PseduoMembran**: Pseudomembrane formation (11 images)
- **OtitExterna**: External ear infection (41 images)
- **Foreign**: Foreign objects in ear (3 images)

### Data Distribution
- **Total Images**: 598 TIFF files (after filtering)
- **Training**: 70% (598 images)
- **Validation**: 15% (127 images)
- **Testing**: 15% (131 images)

### Image Specifications
- **Format**: TIFF files (high quality medical images)
- **Processing**: Resized to 224×224 pixels
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Augmentation**: Random horizontal flip, rotation (±15°)

## Usage Workflow

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete pipeline
python main.py

# 3. View experiment results
mlflow ui

# 4. Use web application
python web/app.py

# 5. Analyze in Jupyter
jupyter notebook notebooks/ear_infection_classifier.ipynb
```

### Pipeline Steps
- **Data Setup**: Download dataset, split into train/val/test
- **Model Training**: Train 6 different fine-tuning strategies
- **Evaluation**: Test all models and select best performer
- **Deployment**: Web interface for real-time predictions

## Technical Architecture

### Model Stack
- **Base Model**: EfficientNetV2-S (pre-trained on ImageNet-21K)
- **Framework**: PyTorch Lightning for clean training loops
- **Metrics**: F1-Score (macro-averaged) for imbalanced medical data
- **Optimization**: Adam optimizer with learning rate scheduling

### Fine-tuning Strategies
1. **freeze_backbone**: Only train classifier head
2. **last1+head**: Train final block + classifier
3. **last2+head**: Train last 2 blocks + classifier
4. **last3+head**: Train last 3 blocks + classifier
5. **last4+head**: Train last 4 blocks + classifier
6. **full**: Train entire model end-to-end

### Experiment Tracking
- **MLflow**: Complete experiment logging and model registry
- **Checkpoints**: Best models automatically saved in `checkpoints/`
- **Metrics**: Training/validation/test performance tracked
- **Artifacts**: Model files, plots, and configurations stored
- **Visualizations**: Confusion matrices saved in `visualizations/`

## Project Organization Benefits

### Clean Structure
- **Organized Data**: All data in `data/` folder with proper subdirectories
- **Clean Root**: No scattered files in main directory
- **Logical Grouping**: Related files grouped together
- **Version Control**: Proper `.gitignore` for generated files

### Standard ML Practices
- **Data Science Structure**: Follows industry best practices
- **Separation of Concerns**: Data, code, experiments, and outputs separated
- **Reproducibility**: All paths and configurations properly managed
- **Scalability**: Easy to extend with new datasets or models

### Development Workflow
- **Modular Code**: Clean separation between data, training, and evaluation
- **Experiment Tracking**: Comprehensive MLflow integration
- **Documentation**: Well-documented code and project structure
- **Testing Ready**: Structure supports easy testing and validation

## File Organization Summary

- **`data/`**: All dataset-related files (raw, processed, splits)
- **`checkpoints/`**: Model checkpoints and training artifacts
- **`visualizations/`**: Generated plots and confusion matrices
- **`mlruns/`**: MLflow experiment tracking data
- **`src/`**: Source code organized by functionality
- **`notebooks/`**: Jupyter notebooks for analysis
- **`web/`**: Web application for deployment
- **`docker/`**: Containerization files

This structure ensures the project is maintainable, scalable, and follows industry best practices for machine learning projects.