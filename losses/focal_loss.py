import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import cross_entropy
# Official implementation of Focal Loss in their paper


class FocalLoss(nn.Module):
    def __init__(self, focus_param=2, balance_param=0.5):
        super(FocalLoss, self).__init__()
        self.focus_param = focus_param
        self.balance_param = balance_param

    def forward(self, inputs, targets):
        log_pt = - F.cross_entropy(inputs, targets)
        pt = torch.exp(log_pt)

        focal_loss = -((1 - pt) ** self.focus_param) * log_pt
        balanced_focal_loss = self.balance_param * focal_loss

        return balanced_focal_loss


def testFocalLoss():
    inputs = Variable(torch.randn(3, 5), requires_grad=True)
    targets = Variable(torch.LongTensor(3).random_(5))
    print(inputs)
    print(targets)
    loss = FocalLoss()
    output = loss(inputs, targets)
    output.backward()
    print(output)
