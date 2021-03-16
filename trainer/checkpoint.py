import torch
import os
import datetime


class CheckPoint:
    """
    - Checkpoint for saving model state
        path (str)
        save_frequency (int): continuously 1 epoch
    """

    def __init__(self, save_per_epoch=1, path=None):
        self.path = path
        self.save_per_epoch = save_per_epoch

        if self.path is None:
            self.path = os.path.join(
                'weights', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    def save(self, model, **kwargs):
        """
        save model and optimizer weight
        :params: pytorch model state dict
        """
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        epoch = kwargs['epoch'] if 'epoch' in kwargs else '0'
        iters = kwargs['iters'] if 'iters' in kwargs else '0'

        model_path = "_".join([model.model_name, str(epoch), str(iters)])

        weights = {
            'model': model.model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        }

        torch.save(weights, os.path.join(self.path, model_path) + '.pt')


def load(model, path):
    """
    - Load checkpoint
        model (nn.Module): model weights
        path (str): checkpoint path
    """
    state = torch.load(path)
    try:
        model.model.load_state_dict(state["model"])
        model.optimizer.load_state_dict(state["optimizer"])

    except KeyError:
        try:
            ret = model.model.load_state_dict(state, strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')

    except torch.nn.modules.module.ModuleAttributeError:
        try:
            ret = model.load_state_dict(state["model"])
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
    print("Loaded Successfully!")
