import torch.nn as nn
from metrics import ClassificationAccuracyMetric


class BaselineModel(nn.Module):
    def __init__(self, optimizer, criterion, metrics=ClassificationAccuracyMetric(), lr=1e-4, freeze=False, device=None, optim_params=None):
        super(BaselineModel, self).__init__()
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.lr = lr
        self.freeze = freeze
        self.device = device
        self.optim_params = optim_params if optim_params is not None else {
            'lr': 1e-3}

        if not isinstance(metrics, list):
            self.metrics = [metrics]

    def set_optimizer_params(self):
        for param in self.optimizer.param_groups:
            for key in param.keys():
                if key in self.optim_params.keys():
                    param[key] = self.optim_params[key]

    def unfreeze(self):
        for params in self.parameters():
            params.requires_grad = True

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
