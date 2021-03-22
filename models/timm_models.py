import timm
import torch.nn as nn


class BaseTimmModel(nn.Module):
    """
    Pretrained models from https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, n_classes, model_arch="vit_base_patch16_384",
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)

        if model_arch.find('nfnet') != -1:
            self.model.head.fc = nn.Linear(
                self.model.head.fc.in_features, n_classes)

        elif model_arch.find('efficientnet') != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, n_classes)

        elif model_arch.find('vit') != -1:
            self.model.head = nn.Linear(self.model.head.in_features, n_classes)

        elif model_arch.find('densenet') != -1:
            self.model.classifier = nn.Linear(
                self.model.classifier.in_features, n_classes)

        elif model_arch.find('resnext') != -1:
            self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

        else:
            assert False, "Classifier block not implemented yet in Timm models"

        self.model = nn.DataParallel(self.model)

    def forward(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs = self.model(inputs)

        return outputs
