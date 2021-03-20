import torch
import torch.nn as nn
from .baseline import BaselineModel
import torchvision.models as models


class Classifier(BaselineModel):
    def __init__(self, backbone, n_classes, **kwargs):
        super(Classifier, self).__init__(**kwargs)
        self.model = backbone
        self.name = 'Classifier'
        self.optimizer = self.optimizer(self.parameters(), lr=self.lr)
        self.set_optimizer_params()
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features,
                                  out_features=self.n_classes)

        if self.device is not None:
            self.model.to(self.device)
            self.criterion.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch["img"]
        targets = batch["category"]
        inputs = torch.FloatTensor(inputs)

        target = [int(i) for i in targets]
        targets = torch.LongTensor(target)
        if self.device is not None:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def inference_step(self, batch):
        inputs = batch['img']
        if self.device:
            inputs = inputs.to(self.device)
        outputs = self(inputs)
        preds = torch.argmax(outputs, dim=1)

        if self.device:
            preds = preds.cpu()
        return preds.numpy()

    def evaluate_step(self, batch):
        inputs = batch["img"]
        targets = batch["category"]
        accuracy = 0

        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        outputs = self(inputs)  # batch_size, category_dim
        loss = self.criterion(outputs, targets)

        accuracy += (outputs == targets).float().sum()
        metric_dict = self.update_metrics(outputs, targets)

        return loss, accuracy, metric_dict

    def forward_test(self):
        inputs = torch.rand(1, 3, 224, 224)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs
