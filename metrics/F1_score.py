import torch
import numpy as np


class ClassificationF1Score:
    def __init__(self, labels, average='macro'):
        self.labels = labels  # int: number of labels
        self.average = average
        self.reset()

    def update(self, outputs, targets):
        pred = torch.argmax(outputs, 1)
        pred = pred.cpu().numpy()
        target = targets.cpu().numpy()
        for pd, gt in zip(pred, target):
            self.count_dict[gt]['total_gt'] += 1
            self.count_dict[pd]['total_p'] += 1
            if pd == gt:
                self.count_dict[pd]['total_pt'] += 1

    def compute(self, item, epsilon=1e-7):
        try:
            precision = item['total_pt']*1.0 / item['total_p']
            recall = item['total_pt']*1.0 / item['total_gt']
        except ZeroDivisionError:
            return 0

        if precision + recall > 0:
            score = 2*precision*recall / (precision + recall + epsilon)
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

        return {'f1-score': score}

    def __str__(self) -> str:
        return f'F1-score: {self.value()}'


if __name__ == '__main__':
    f1 = ClassificationF1Score(3)
    y_true = torch.tensor([0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([[3, 1, 2], [0, 1, 2], [1, 3, 1], [
                          3, 2, 0], [3, 1, 1], [2, 3, 1]])
    f1.update(y_pred, y_true)
    print(f1.value())
