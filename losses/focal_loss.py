import torch
import torch.nn as nn
import torch.nn.functional as F

# Official implementation of Focal Loss in their paper


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(input=inputs,
                                                          target=target,
                                                          reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(input=inputs,
                                              target=target,
                                              reduce=False)

        p_t = torch.exp(BCE_loss)

        focal_loss = self.alpha*(1-p_t)**self.gamma*BCE_loss

        if self.reduce:
            return torch.mean(focal_loss)

        return focal_loss
