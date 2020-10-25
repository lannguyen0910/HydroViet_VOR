import torch


class ClassificationAccuracyMetric:
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
        return self.correct/self.output_size

    def __str__(self) -> str:
        return f'Accuracy: {self.value()}'

    def __len__(self):
        return self.output_size
