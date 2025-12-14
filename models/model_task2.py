import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import F1Score, Accuracy


class PhonemeModel(nn.Module):
    """Simple Conv model for phoneme classification"""
    def __init__(self, input_dim, model_dim, num_classes, dropout_rate=0.3, lstm_layers=1, bi_directional=True, batch_norm=False):
        super().__init__()
        self.num_classes = num_classes
        
        # 2-layer Conv with BatchNorm and Global Average Pooling
        self.model = nn.Sequential(
            # First conv layer
            nn.Conv1d(input_dim, model_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second conv layer
            nn.Conv1d(model_dim, model_dim * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(model_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            
            # Classifier
            nn.Linear(model_dim * 2, model_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(model_dim, num_classes)
        )

    def forward(self, x):
        return self.model(x)


class BrainPhonemeClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes=40,  # Base phonemes (without position markers)
        lr=5e-4,
        weight_decay=0.0,
        dropout_rate=0.3,
        label_smoothing=0.0,
        batch_norm=False,
        lstm_layers=1,
        bi_directional=False,
        phoneme_to_idx=None,  # Mapping from phoneme to class index
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['phoneme_to_idx'])
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        
        # Adaptive loss weights from competition baseline
        self.class_weights = self._get_adaptive_weights(phoneme_to_idx, num_classes)

        self.model = PhonemeModel(
            input_dim=input_dim,
            model_dim=model_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            lstm_layers=lstm_layers,
            bi_directional=bi_directional,
            batch_norm=batch_norm,
        )

        # CrossEntropyLoss with adaptive weights and label smoothing
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=label_smoothing
        )

        # Metrics for multiclass classification
        self.train_f1 = F1Score(
            task='multiclass', average='macro', num_classes=num_classes)
        self.val_f1 = F1Score(
            task='multiclass', average='macro', num_classes=num_classes)
        self.test_f1 = F1Score(
            task='multiclass', average='macro', num_classes=num_classes)
        
        self.train_acc = Accuracy(
            task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(
            task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(
            task='multiclass', num_classes=num_classes)
        
        self.train_loss_total = 0.0
        self.train_loss_count = 0
        self.val_loss_total = 0.0
        self.val_loss_count = 0

    def _get_adaptive_weights(self, phoneme_to_idx, num_classes):
        """Get adaptive loss weights based on competition baseline"""
        # Default weight is 1.0 for all classes
        weights = torch.ones(num_classes)
        
        if phoneme_to_idx is None:
            return weights
        
        # Adaptive weights from Table 2 in competition baseline
        # These are for base phonemes (without position markers)
        phoneme_weights = {
            'ey': 0.05,
            'ay': 3.00,
            'uh': 10.00,
            'uw': 3.00,
            's': 0.80,
            'sh': 3.00,
            'm': 3.00,
            'ae': 3.00,
            'jh': 1.50,
            'ah': 2.00,
        }
        
        # Map phoneme weights to class indices
        for phoneme, weight in phoneme_weights.items():
            if phoneme in phoneme_to_idx:
                idx = phoneme_to_idx[phoneme]
                weights[idx] = weight
        
        print(f"[INFO] Applied adaptive weights to {len(phoneme_weights)} phoneme classes")
        return weights

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)  # (B, num_classes)
        loss = self.loss_fn(logits, y.long())
        preds = torch.argmax(logits, dim=1)

        f1_metric = getattr(self, f"{stage}_f1")
        acc_metric = getattr(self, f"{stage}_acc")
        f1_metric.update(preds, y.long())
        acc_metric.update(preds, y.long())

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

        if stage == "train":
            self.train_loss_total += loss.detach().item()
            self.train_loss_count += 1
        elif stage == "val":
            self.val_loss_total += loss.detach().item()
            self.val_loss_count += 1

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log("train_f1_macro", self.train_f1.compute(), prog_bar=True)
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.train_f1.reset()
        self.train_acc.reset()

        if self.train_loss_count > 0:
            avg_loss = self.train_loss_total / self.train_loss_count
            self.log("train_loss", avg_loss, prog_bar=True)
    def on_validation_epoch_end(self):
        self.log("val_f1_macro", self.val_f1.compute(), prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.val_f1.reset()
        self.val_acc.reset()

        if self.val_loss_count > 0:
            avg_loss = self.val_loss_total / self.val_loss_count
            self.log("val_loss", avg_loss, prog_bar=True)
            self.val_loss_total = 0.0
            self.val_loss_count = 0

    def on_test_epoch_end(self):
        self.log("test_f1_macro", self.test_f1.compute(), prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        self.test_f1.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        # AdamW with Cosine Annealing for better convergence
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,      # Initial restart period
            T_mult=2,    # Period multiplier
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
    def configure_optimizers(self):
        # Use simple Adam optimizer like baseline
        return torch.optim.Adam(self.parameters(), lr=self.lr)
