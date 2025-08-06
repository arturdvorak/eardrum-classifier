Eardrum Classification with EfficientNetV2 + PyTorch Lightning

This project builds a robust classifier for tympanic membrane (eardrum) conditions using medical images. It leverages a pre-trained EfficientNetV2-S model from timm, fine-tunes it with PyTorch Lightning, and evaluates multiple transfer learning strategies.

⸻

Dataset
	•	Source: Kaggle – Eardrum Dataset (Otitis Media)
	•	Total images: 956
	•	Classes Used:
	•	Normal
	•	Acute Otitis Media (AOM)
	•	Chronic Otitis Media
	•	Otitis Externa
	•	Earwax
	•	Tympanosclerosis
	•	Excluded classes (due to very low samples):
	•	Foreign
	•	PseduoMembran
	•	Earventulation

Later, for Cycle 2, only 4 classes were retained:
	•	Aom
	•	Chronic
	•	Earwax
	•	Normal

⸻

Project Pipeline

1. Dataset Setup
	•	Download from Kaggle
	•	Unpack & clean old versions
	•	Remove underrepresented classes
	•	Split into train/, val/, and test/ folders (70/15/15)

2. Data Preprocessing
	•	Resize images to 224×224
	•	Augment training images (horizontal flip, rotation)
	•	Normalize using ImageNet mean/std
	•	Load with ImageFolder and wrap with DataLoader

3. Model: EfficientNetV2-S
	•	From timm: tf_efficientnetv2_s.in21k
	•	Final classification head replaced to match number of classes

4. Evaluation Metric
	•	Primary: Macro F1 Score (robust to class imbalance)

5. Lightning Module

Defines:
	•	forward(), training_step(), validation_step(), test_step()
	•	configure_optimizers() with AdamW + ReduceLROnPlateau
	•	torchmetrics.MulticlassF1Score for validation/test

6. Fine-Tuning Strategies

Each trained with EarlyStopping and ModelCheckpoint:
	•	freeze_backbone: Only head is trainable
	•	last1+head, last2+head, last3+head, last4+head: Gradually unfreeze blocks
	•	full: All layers trainable

Tracked via MLflow:
	•	Metrics, parameters, training curves
	•	Best checkpoints
	•	Suggested learning rates (via Lightning Tuner)

7. Model Evaluation (on test set)
	•	Load best checkpoint per strategy
	•	Run predictions on test data
	•	Compute accuracy, precision, recall, F1
	•	Display confusion matrix
	•	Log metrics and classification report to MLflow

8. Strategy Comparison
	•	Compare evaluation metrics across strategies
	•	Tabulate accuracy, precision, recall, F1

9. Model Improvement
	•	Integrate learning rate tuning via Lightning’s Tuner
	•	Re-train each strategy with optimal LR
	•	Re-evaluate and update strategy summary

10. Per-Class Performance
	•	Compute and visualize F1 Score per class
	•	Analyze model strengths and weaknesses

11. Cycle 2: Reduced Class Training
	•	New training cycle with 4 main classes (Aom, Chronic, Earwax, Normal)
	•	Repeat full training & evaluation pipeline

12. Export Best Model
	•	Save top-performing checkpoint
	•	Upload to Google Drive for sharing/deployment

⸻

Streamlit App

A simple web interface to upload an eardrum image and get predictions.
Optional webhook integration for tools like Microsoft Flow or other automation platforms.

⸻

Main Libraries
	•	PyTorch
	•	torchvision
	•	timm
	•	PyTorch Lightning
	•	torchmetrics
	•	scikit-learn
	•	mlflow
	•	seaborn
	•	streamlit