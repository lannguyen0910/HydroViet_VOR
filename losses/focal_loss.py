from utils.techniques.one_hot import *
from torch.autograd import Variable
from torch.nn.functional import cross_entropy
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
sys.path.append('../')

# Official implementation of Focal Loss in their paper


class FocalLoss1(nn.Module):
    def __init__(self, focus_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()
        self.focus_param = focus_param
        self.balance_param = balance_param

    def forward(self, inputs, targets):
        log_pt = - F.cross_entropy(inputs, targets)
        pt = torch.exp(log_pt)

        focal_loss = -((1 - pt) ** self.focus_param) * log_pt
        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


class FocalLoss2(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss

        return loss.sum()


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        # weight parameter will act as the alpha parameter to balance class weights
        self.weight = weight
        self.gamma = gamma

    def forward(self, input, target):
        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


def testFocalLoss():
    inputs = Variable(torch.randn(3, 5), requires_grad=True)
    targets = Variable(torch.LongTensor(3).random_(5))
    print(inputs)
    print(targets)
    loss = FocalLoss()
    loss1 = FocalLoss1()
    loss2 = FocalLoss2()

    output = loss(inputs, targets)
    output1 = loss1(inputs, targets)
    output2 = loss2(inputs, targets)

    # output.backward()
    print(output)
    print(output1)
    print(output2)
