import torch
import os
import datetime


class CheckPoint:
    """
    - Checkpoint for saving model state
        path (str)
        save_frequency (int): continuously 1 epoch
    """

    def __init__(self, save_per_epoch=1, path='weights'):
        self.path = path
        self.save_per_epoch = save_per_epoch

        self.path = os.path.join(
            path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    def save(self, model, **kwargs):
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        epoch = kwargs['epoch'] if 'epoch' in kwargs else '0'
        model_path = "_".join([model.model_name, str(epoch)])

        weights = {
            'model': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }

        torch.save(weights, os.path.join(self.path, model_path) + '.pt')


def load(model, path):
    """
    - Load checkpoint
        model (.pt): model weights
        path (str): checkpoint path
    """
    state = torch.load(path)
    model.model.load_state_dict(state['model'])
    model.optimizer.load_state_dict(state['optimizer'])
    print('Loaded!')
