import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from metrics.metrics_template import TemplateMetric


class ClassificationF1Score(TemplateMetric):
    """
    - F1-score in image classification (types: 'macro', 'micro', 'weighted')
    """

    def __init__(self, labels, average='macro'):
        self.labels = labels  # int: number of labels
        self.average = average
        self.reset()

    def update(self, outputs, targets):
        # print('Output: ', outputs)
        # print('Target: ', targets)
        pred = torch.argmax(outputs, 1)
        # print('Output after argmax: ', pred)
        pred = pred.cpu().numpy()
        target = targets.cpu().numpy()
        # print('Target shape: ', target.shape)

        for pd, gt in zip(pred, target):
            # print('Pd: ', pd)
            # print('Gt: ', gt)
            self.count_dict[gt]['total_gt'] += 1
            self.count_dict[pd]['total_p'] += 1
            if pd == gt:
                self.count_dict[pd]['total_pt'] += 1

        print(self.count_dict)

    def compute(self, item, epsilon=1e-7):
        try:
            precision = item['total_pt']*1.0 / item['total_p']
            recall = item['total_pt']*1.0 / item['total_gt']
        except ZeroDivisionError:
            return 0

        if precision + recall > 0:
            score = 2 * precision * recall / (precision + recall + epsilon)
        else:
            score = 0

        return score

    def compute_micro(self):
        total_p = sum([self.count_dict[label]['total_p']
                       for label in range(self.labels)])
        total_gt = sum([self.count_dict[label]['total_gt']
                        for label in range(self.labels)])
        total_pt = sum([self.count_dict[label]['total_pt']
                        for label in range(self.labels)])

        return self.compute({
            'total_p': total_p,
            'total_gt': total_gt,
            'total_pt': total_pt
        })

    def compute_macro(self):
        results = [self.compute(self.count_dict[label])
                   for label in range(self.labels)]
        results = sum(results)*1.0 / self.labels
        return results

    def compute_weighted(self):
        weights = [abs(random.random()) for _ in range(self.labels - 1)]
        weights = weights + [1 - sum(weights)]

        results = [self.compute(self.count_dict[label] * weight)
                   for label, weight in zip(range(self.labels), weights)]

        results = sum(results*1.0) / self.labels

        return results

    def reset(self):
        self.count_dict = {
            label: {
                'total_gt': 0,
                'total_p': 0,
                'total_pt': 0
            } for label in range(self.labels)
        }

    def value(self):
        if self.average == 'micro':
            score = self.compute_micro()
        elif self.average == 'macro':
            score = self.compute_macro()
        elif self.average == 'weighted':
            score = self.compute_weighted()

        return {'f1-score': score}

    def __str__(self) -> str:
        return f'F1-score: {self.value()}'


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()


def test():
    f1 = ClassificationF1Score(3)
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([[3, 1, 2], [0, 1, 2], [1, 3, 1], [
                          3, 2, 0], [3, 1, 1], [2, 3, 1]])
    f1.update(y_pred, y_true)
    print(f1.value())
