import torch
import numpy as np


def compute_multiclass(output, target):
    pred = torch.argmax(output, dim=1)
    correct = (pred == target).sum()
    sample_size = output.size(0)
    return correct, sample_size


def compute_binary(output, target, threshold=0.75):
    pred = output >= threshold
    correct = (pred == target).sum()
    sample_size = output.size(0)
    return correct, sample_size


def get_compute(types=None):
    if types == 'binary':
        return compute_binary

    elif types == 'multi':
        return compute_multiclass


class ClassificationAccuracyMetric:
    """
    - Simple classification metrics to compute accuracy base on number of correct outputs
    """

    def __init__(self, types='multi', decimals=6):
        self.decimals = decimals
        self.compute_fn = get_compute(types)
        self.reset()

    def compute(self, outputs, targets):
        return self.compute_fn(outputs.squeeze(0), targets)

    def update(self, outputs, targets):
        assert isinstance(outputs, torch.Tensor), 'Ouput must be a tensor'
        value = self.compute(outputs, targets)

        self.correct += value[0]
        self.output_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.output_size = 0

    def value(self):
        values = self.correct / self.output_size
        if values.is_cuda:
            values = values.cpu()

        return {'Accuracy: ': np.around(values.numpy(), decimals=self.decimals)}

    def __str__(self) -> str:
        return f'Accuracy: {self.value()}'

    def __len__(self):
        return self.output_size


def test():
    accuracy = ClassificationAccuracyMetric()
    output = [[1, 4, 2],
              [5, 7, 4],
              [2, 3, 0]]
    target = [[1, 1, 0]]
    output = torch.LongTensor(output)
    target = torch.LongTensor(target)

    accuracy.update(output, target)
    dic = {}
    dic.update(accuracy.value())
    print(dic)


if __name__ == '__main__':
    test()

    # Output:  torch.Size([3, 3])
    # Target:  torch.Size([1, 3])
    # Values:  tensor(0.6667)
    # {'Accuracy: ': 0.666667}
