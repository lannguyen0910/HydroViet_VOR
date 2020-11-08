import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Logger():
    """
    Logger Tensorboard
        - log_dir (str): path to save logs
    """

    def __init__(self, log_dir=None):
        self.log_dir = log_dir

        if self.log_dir is None:
            self.log_dir = os.path.join(
                'loggers/run', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.iters = {}

    def write(self, tags, values):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        """
        if not isinstance(tags, list):
            tags = list(tags)
        if not isinstance(values, list):
            values = list(values)

        for i, (tag, value) in enumerate(zip(tags, values)):
            if tag not in self.iters.keys():
                self.iters[tag] = 0
            self.writer.add_scalar(tag, value, self.iters[tag])
            self.iters[tag] += 1
