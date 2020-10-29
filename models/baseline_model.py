import torch
import torch.nn as nn
from metrics import ClassificationAccuracyMetric


class BaselineModel(nn.Module):
    def __init__(self, criterion, metrics=ClassificationAccuracyMetric(), n_labels=None, model=None, lr=1e-4, freeze=False, device=None):
        super().__init__()
        # self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.lr = lr
        self.freeze = freeze
        self.device = device
        self.model = model
        self.n_labels = n_labels

        if device is not None:
            self.criterion.to(device)

        if not isinstance(metrics, list):
            self.metrics = [metrics]

    def modify_last_layer(self):
        if self.model is not None:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features=in_features,
                                      out_features=self.n_labels)

            self.optimizer = torch.optim.Adam(
                self.model.fc.parameters(), lr=self.lr)

            if self.device is not None:
                self.model.fc.to(self.device)

    def trainable_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) \
            if self.model is not None else sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freezing(self):
        if self.freeze and self.model is not None:
            for params in self.model.parameters():
                params.requires_grad = False

    def unfreeze(self):
        if self.model is not None:
            for params in self.model.parameters():
                params.requires_grad = True
        else:
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
