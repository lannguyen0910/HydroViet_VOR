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
from configs.config import Hyperparams as hp
from configs.config import Config

from utils.cuda import *

import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d or torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)


def get_lr_policy(opt_config):
    optimizer_params = {}
    lr = opt_config['lr'] if 'lr' in opt_config.keys() else None
    if opt_config["name"] == 'sgd':
        optimizer = SGD
        optimizer_params = {
            'lr': lr,
            'weight_decay': opt_config['weight_decay'],
            'momentum': opt_config['momentum'],
            'nesterov': True}
    elif opt_config["name"] == 'adam':
        optimizer = AdamW
        optimizer_params = {
            'lr': lr,
            'weight_decay': opt_config['weight_decay'],
            'betas': (opt_config['momentum'], 0.999)}
    return optimizer, optimizer_params


def get_lr_scheduler(optimizer, lr_config, **kwargs):

    scheduler_name = lr_config["name"]
    step_per_epoch = False

    if scheduler_name == '1cycle-yolo':
        def one_cycle(y1=0.0, y2=1.0, steps=100):
            # lambda function for sinusoidal ramp from y1 to y2
            return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1

        lf = one_cycle(1, 0.158, kwargs['num_epochs'])  # cosine 1->hyp['lrf']
        scheduler = LambdaLR(optimizer, lr_lambda=lf)
        step_per_epoch = True

    elif scheduler_name == '1cycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            epochs=kwargs['num_epochs'],
            steps_per_epoch=int(
                len(kwargs["trainset"]) / kwargs["batch_size"]),
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=10**5)
        step_per_epoch = False

    elif scheduler_name == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=1,
            verbose=False,
            threshold=0.0001,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-8,
            eps=1e-08
        )
        step_per_epoch = True

    elif scheduler_name == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs['num_epochs'],
            T_mult=1,
            eta_min=0.0001,
            last_epoch=-1,
            verbose=False
        )
        step_per_epoch = False
    return scheduler, step_per_epoch


def get_dataset_and_dataloader(config):

    train_transforms = get_augmentation(config, type='train')
    val_transforms = get_augmentation(config, type='val')

    trainset = ImageClassificationDataset(
        root=os.path.join(config.data, config.project_name, config.train_imgs),
        transforms=train_transforms)

    valset = ImageClassificationDataset(
        root=os.path.join(
            config.data, config.project_name, config.val_imgs),
        transforms=val_transforms)

    trainloader = data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=trainset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True)

    valloader = data.DataLoader(
        valset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=valset.collate_fn,
        num_workers=config.num_workers,
        pin_memory=True)

    return trainset, valset, trainloader, valloader
