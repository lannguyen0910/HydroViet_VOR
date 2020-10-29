import torch
import torch.nn as nn
from metrics import ClassificationAccuracyMetric


class BaselineModel:
    def __init__(self, optimizer, criterion, metrics=ClassificationAccuracyMetric(), lr=1e-4, freeze=False, device=None):
        super().__init__()
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.lr = lr
        self.freeze = freeze
        self.device = device

        if device is not None:
            self.criterion.to(device)

        if not isinstance(metrics, list):
            self.metrics = [metrics]

    def trainable_parameters(self):
        return sum(p.numel for p in self.parameters() if p.requires_grad)

    def unfreeze(self):
        for params in self.parameters():
            params.requires_grad = True

    def update_metrics(self, outputs, targets):
        metric_dict = {}
        for metric in self.metrics:
            metric.update(outputs, targets)
            items = {metric: metric.value()}
            metric_dict.update(items)

        return metric_dict

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()
