import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import F1Score
from utils.loss import BCEWithLogitsLossWithSmoothing

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        # x: (B, T, 2D)
        attn_scores = self.attn(x)                      # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # softmax over time
        context = (x * attn_weights).sum(dim=1)         # weighted sum over T
        return context  # (B, 2D)

class SpeechModel(nn.Module):
    def __init__(self, input_dim, model_dim, dropout_rate=0.3, lstm_layers=1, bi_directional=True, batch_norm=False):
        super().__init__()
        self.model_dim = model_dim
        self.bi_directional = bi_directional
        self.lstm_layers = lstm_layers

        self.conv = nn.Conv1d(input_dim, model_dim, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(model_dim) if batch_norm else nn.Identity()
        self.conv_dropout = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bi_directional,
        )

        lstm_output_dim = model_dim * 2 if bi_directional else model_dim
        self.attention = AttentionPooling(input_dim=lstm_output_dim)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        self.speech_classifier = nn.Linear(lstm_output_dim, 1)

    def forward(self, x):
        x = self.conv(x)                      # (B, D, T)
        x = self.batch_norm(x)
        x = self.conv_dropout(x)
        x = x.permute(0, 2, 1)                # (B, T, D)

        lstm_out, _ = self.lstm(x)            # (B, T, 2D)
        attn_out = self.attention(lstm_out)   # (B, 2D)

        x = self.lstm_dropout(attn_out)
        x = self.speech_classifier(x)         # (B, 1)
        return x

class BrainSpeechClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes=1,
        lr=5e-5,
        weight_decay=0.01,
        dropout_rate=0.3,
        smoothing=0.0,
        pos_weight=0.5,
        batch_norm=False,
        lstm_layers=1,
        bi_directional=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.smoothing = smoothing
        self.pos_weight = pos_weight

        self.model = SpeechModel(
            input_dim=input_dim,
            model_dim=model_dim,
            dropout_rate=dropout_rate,
            lstm_layers=lstm_layers,
            bi_directional=bi_directional,
            batch_norm=batch_norm,
        )

        self.loss_fn = BCEWithLogitsLossWithSmoothing(smoothing=smoothing, pos_weight=pos_weight)

        self.train_f1 = F1Score(
            task='multiclass', average='macro', num_classes=2)
        self.val_f1 = F1Score(
            task='multiclass', average='macro', num_classes=2)
        self.test_f1 = F1Score(
            task='multiclass', average='macro', num_classes=2)

        self.train_loss_total = 0.0
        self.train_loss_count = 0
        self.val_loss_total = 0.0
        self.val_loss_count = 0

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y.unsqueeze(1).float())
        preds = (torch.sigmoid(logits) > 0.5).long().squeeze()

        f1_metric = getattr(self, f"{stage}_f1")
        f1_metric.update(preds, y.long())

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
        self.train_f1.reset()

        if self.train_loss_count > 0:
            avg_loss = self.train_loss_total / self.train_loss_count
            self.log("train_loss", avg_loss, prog_bar=True)
            self.train_loss_total = 0.0
            self.train_loss_count = 0

    def on_validation_epoch_end(self):
        self.log("val_f1_macro", self.val_f1.compute(), prog_bar=True)
        self.val_f1.reset()

        if self.val_loss_count > 0:
            avg_loss = self.val_loss_total / self.val_loss_count
            self.log("val_loss", avg_loss, prog_bar=True)
            self.val_loss_total = 0.0
            self.val_loss_count = 0

    def on_test_epoch_end(self):
        self.log("test_f1_macro", self.test_f1.compute(), prog_bar=True)
        self.test_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1_macro',
                'interval': 'epoch',
                'frequency': 1
            }
        }