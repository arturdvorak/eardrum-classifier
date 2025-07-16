# model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from torchmetrics.classification import MulticlassF1Score

class EfficientNetV2Lightning(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model("tf_efficientnetv2_s.in21k", pretrained=True, num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average='macro')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        f1 = self.val_f1(preds, y)
        self.log("val_loss", loss)
        self.log("val_f1", f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        f1 = self.test_f1(preds, y)
        self.log("test_loss", loss)
        self.log("test_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_f1",  # Monitors F1 to reduce LR
                "interval": "epoch",
                "frequency": 1
            }
        }