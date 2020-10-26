import torch
import os


class CheckPoint:
    def __init__(self, save_per_epoch=1, path='weights'):
        self.path = path
        self.save_per_epoch = save_per_epoch

        if not os.path.exists(path):
            os.mkdir(path)

    def save(self, model, **kwargs):
        epoch = kwargs['epoch'] if 'epoch' in kwargs else '0'
        model_path = "_".join([model.model_name, str(epoch)])

        weights = {
            'model': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }

        torch.save(weights, os.path.join(self.path, model_path) + '.pt')


def load(model, path):
    state = torch.load(path)
    model.model.load_state_dict(state['model'])
    model.optimizer.load_state_dict(state['optimizer'])
    print('Loaded!')