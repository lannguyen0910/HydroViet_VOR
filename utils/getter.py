from metrics import *
from losses import *
from datasets import *
from dataloaders import *
from models import *
from trainer import *
from augmentation import *
from logger import *
from utils.helper import *
from .meter import *
# from utils.split_data import *
# from ssd import *
from configs.config import Hyperparams as hp
from configs.config import Config

import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d or torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum() * 1.0/dista.size()[0]


def get_instance(config, **kwargs):
    assert 'name' in config
    config.setdefault('args', {})
    if config['args'] is None:
        config['args'] = {}
    return globals()[config['name']](**config['args'], **kwargs)
