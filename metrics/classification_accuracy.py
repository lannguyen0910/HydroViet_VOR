import torch
import numpy as np


class ClassificationAccuracyMetric:
    """
    - Simple classification metrics to compute accuracy base on number of correct outputs
    """

    def __init__(self, decimals=6):
        self.decimals = decimals
        self.reset()

    def calculate(self, output, target):
        output_size = output.size(0)
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum()
        return correct, output_size

    def update(self, output, target):
        assert isinstance(output, torch.Tensor), 'Ouput must be a tensor'
        value = self.calculate(output, target)

        self.correct += value[0]
        self.output_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.output_size = 0

    def value(self):
        values = self.correct / self.output_size
        print('Values: ', values)
        if values.is_cuda:
            values = values.cpu()
        return {'Accuracy: ': np.around(values.numpy(), decimals=self.decimals)}

    def __str__(self) -> str:
        return f'Accuracy: {self.value()}'

    def __len__(self):
        return self.output_size


if __name__ == '__main__':
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
