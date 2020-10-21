import torch


class ClassificationAccuracyMetric():
    def __init__(self, *args, **kwargs) -> None:
        self.reset()

    def calculate(self, output, target):
        output_size = output.size(0)
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum()
        return correct, output_size

    def update(self, value):
        self.correct += value[0]
        self.output_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.output_size = 0.0

    def value(self):
        return self.correct/self.output_size

    def summary(self):
        print(f'Accuracy: {self.value()}')
