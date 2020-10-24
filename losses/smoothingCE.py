import torch
import torch.nn as nn


class SmoothingCE(nn.Module):
    def __init__(self, alpha=1e-6, ignore_index=None, reduction='mean'):
        super(SmoothingCE, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, outputs, targets):
        batch_size, n_classes = outputs.shape
        y_hot = torch.zeros(outputs.shape).scatter_(
            1, targets.unsqueeze(1), 1.0)
        y_smooth = (1 - self.alpha) * y_hot + self.alpha / n_classes
        loss = torch.sum(- y_smooth *
                         nn.functional.log_softmax(outputs, -1), -1).sum()
        if self.reduction:
            loss /= batch_size

        return loss
