class TemplateMetric():
    """
    Accuracy for Classification
    """

    def __init__(self, decimals=10):
        self.reset()

    def compute(self, output, target):
        pass

    def update(self, output, target):
        pass

    def reset(self):
        self.sample_size = 0
        pass

    def value(self):
        return {'None': None}

    def __str__(self):
        return f'None: {self.value()}'

    def __len__(self):
        return len(self.sample_size)
