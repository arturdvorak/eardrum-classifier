<details>
<summary>Click to expand and copy the raw Markdown code</summary>

# Eardrum Classification with EfficientNetV2 + PyTorch Lightning

This project builds a robust classifier for tympanic membrane (eardrum) conditions using medical images. It leverages a pre-trained EfficientNetV2-S model from `timm`, fine-tunes it with PyTorch Lightning, and evaluates multiple transfer learning strategies.

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

## Project Pipeline

### Dataset Setup

- Download from Kaggle  
- Unpack & clean old versions  
- Remove underrepresented classes  
- Split into `train/`, `val/`, and `test/` folders (70/15/15)

### Data Preprocessing

- Resize images to `224×224`  
- Augment training images (horizontal flip, rotation)  
- Normalize using ImageNet mean/std  
- Load with `ImageFolder` and wrap with `DataLoader`

### Model: EfficientNetV2-S

- From `timm`: `tf_efficientnetv2_s.in21k`  
- Final classification head replaced to match number of classes

### Evaluation Metric

- **Primary**: Macro F1 Score (robust to class imbalance)

### Lightning Module

Defines:

- `forward()`, `training_step()`, `validation_step()`, `test_step()`  
- `configure_optimizers()` with AdamW + ReduceLROnPlateau  
- `torchmetrics.MulticlassF1Score` for validation/test  

### Fine-Tuning Strategies

Each trained with `EarlyStopping` and `ModelCheckpoint`:

- `freeze_backbone`: Only head is trainable  
- `last1+head`, `last2+head`, `last3+head`, `last4+head`: Gradually unfreeze blocks  
- `full`: All layers trainable

Tracked via **MLflow**:

- Metrics, parameters, training curves  
- Best checkpoints  
- Suggested learning rates (via Lightning Tuner)

### Model Evaluation (on test set)

- Load best checkpoint per strategy  
- Run predictions on test data  
- Compute accuracy, precision, recall, F1  
- Display confusion matrix  
- Log metrics and classification report to MLflow  

### Strategy Comparison

- Compare evaluation metrics across strategies  
- Tabulate accuracy, precision, recall, F1  

### Model Improvement

- Integrate learning rate tuning via Lightning’s Tuner  
- Re-train each strategy with optimal LR  
- Re-evaluate and update strategy summary  

### Per-Class Performance

- Compute and visualize F1 Score per class  
- Analyze model strengths and weaknesses  

### Cycle 2: Reduced Class Training

- New training cycle with 4 main classes (Aom, Chronic, Earwax, Normal)  
- Repeat full training & evaluation pipeline  

### Export Best Model

- Save top-performing checkpoint  
- Upload to Google Drive for sharing/deployment

---

## Streamlit App

A simple web interface to upload an eardrum image and get predictions.  
Optional webhook integration for tools like Microsoft Flow or other automation platforms.

---

## Main Libraries

- PyTorch  
- torchvision  
- timm  
- PyTorch Lightning  
- torchmetrics  
- scikit-learn  
- mlflow  
- seaborn  
- streamlit  

</details>