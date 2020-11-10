import torch
from torch.autograd import Variable


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def one_hot(labels, classes):
    size = labels.size() + (classes,)
    view = labels.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    labels = labels.view(*view)
    ones = 1.

    if isinstance(labels, Variable):
        ones = Variable(torch.Tensor(labels.size()).fill_(1))
        mask = Variable(mask, volatile=labels.volatile)

    return mask.scatter_(1, labels, ones)
