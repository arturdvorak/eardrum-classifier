Eardrum Classification with EfficientNetV2 + PyTorch Lightning

This project builds a robust classifier for tympanic membrane (eardrum) conditions using medical images. It leverages a pre-trained EfficientNetV2-S model from timm, fine-tunes it with PyTorch Lightning, and evaluates multiple transfer learning strategies.

⸻

Dataset
	•	Source: Kaggle - Eardrum Dataset (Otitis Media)
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

⸻

Project Pipeline

1. Dataset Setup
	•	Download from Kaggle
	•	Unpack & clean up previous versions
	•	Remove underrepresented classes
	•	Split into train/, val/, and test/ (70/15/15)

2. Data Preprocessing
	•	Resize to 224×224
	•	Train set uses augmentation (flip, rotation)
	•	Normalization using ImageNet stats
	•	Loaded with ImageFolder and DataLoader

3. Model: EfficientNetV2-S
	•	From timm: tf_efficientnetv2_s.in21k
	•	Final classification head replaced to match number of classes

4. Evaluation Metric
	•	Primary: Macro F1 Score (to handle class imbalance)

5. Lightning Module

Defines:
	•	forward(), training_step(), validation_step(), test_step()
	•	Loss: CrossEntropyLoss
	•	Metrics: torchmetrics.MulticlassF1Score
	•	Optimizer: AdamW with ReduceLROnPlateau scheduler

6. Fine-Tuning Strategies

Each trained with early stopping & checkpointing:
	•	freeze_backbone: Only train classification head
	•	last1+head → last4+head: Unfreeze last N blocks + head
	•	full: All layers trainable

Logged using MLflow:
	•	Training curves
	•	Best checkpoints
	•	Suggested learning rates from Lightning’s tuner

7. Model Evaluation (on test set)
	•	Load best checkpoint per strategy
	•	Run predictions on test data
	•	Compute final metrics
	•	Display confusion matrix
	•	Log to MLflow with classification report

8. Streamlit App

A simple web interface to upload an eardrum image and get predictions. Optional webhook support to send results to Microsoft Flow or other automation tools.


Main libraries:
	•	PyTorch
	•	torchvision
	•	timm
	•	PyTorch Lightning
	•	torchmetrics
	•	scikit-learn
	•	mlflow
	•	seaborn
	•	streamlit