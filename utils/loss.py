import torch 
import torch.nn as nn

class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=1):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, logits, targets):
        targets = targets.float()
        smoothed = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, smoothed)