import torch
import torch.nn as nn
from .baseline import BaselineModel


class Classifier(BaselineModel):
    def __init__(self, model, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.model = model

        if self.optimizer is not None:
            self.optimizer = self.optimizer(self.parameters(), lr=self.lr)
            self.set_optimizer_params()

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        if self.device is not None:
            self.model.to(self.device)
            self.criterion.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):

        outputs = self.model(batch, self.device)
        categories = batch['categories'].to(self.device)

        loss = self.criterion(outputs, categories)
        loss_dict = {'T': loss.item()}

        return loss, loss_dict

    def inference_step(self, batch):
        outputs = self.model(batch, self.device)
        preds = torch.argmax(outputs, dim=1)
        if self.device:
            preds = preds.detach().cpu()
        return preds.numpy()

    def evaluate_step(self, batch):
        outputs = self.model(batch, self.device)
        targets = batch["category"].to(self.device)

        loss = self.criterion(outputs, targets)
        metric_dict = self.update_metrics(outputs, targets)

        return loss, metric_dict

    def forward_test(self):
        inputs = torch.rand(1, 3, 224, 224)

        if self.device:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs
