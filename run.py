from models.classifier import Classifier
from random import shuffle
from numpy.lib.npyio import save
from torchvision import transforms
from torchvision.models.resnet import ResNet,  resnet34
from tqdm import tqdm
from datasets.transform import transforming
from datasets.image_classification import ImageClassificationDataset
from utils.getter import *

import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# from torchsummary import summary

img_size = (224, 224)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
transform_train = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_val = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    normalize
])
if __name__ == '__main__':
    train_set = ImageClassificationDataset(root='trainingSet/trainingSet',
                                           transforms=transform_train, shuffle=True)
    val_set = ImageClassificationDataset(root='trainingSample',
                                         transforms=transform_val,  shuffle=True)
    print('Train set: ', train_set)
    print('Val set: ', val_set)
    N_CATEGORIES = len(train_set.n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    BATCH_SIZE = 32
    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE)

    EPOCHS = 10
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    metrics = [ClassificationF1Score(
        N_CATEGORIES, average='macro'), ClassificationAccuracyMetric(decimals=3)]
    optimizer = torch.optim.Adam

    resnet = resnet34(pretrained=True)
    model = Classifier(backbone=resnet, n_classes=N_CATEGORIES, optimizer=optimizer, criterion=criterion, metrics=metrics,
                       lr=1e-4, freeze=True, device=None, optim_params=None)
    model.modify_last_layer()
    print('Number of trainable parameters in model: ',
          model.trainable_parameters())

    chpoint = CheckPoint(save_per_epoch=2)

    trainer = Trainer(model, train_loader, val_loader,
                      checkpoint=chpoint, evaluate_per_epoch=2)
    trainer.fit(num_epochs=EPOCHS, print_per_iter=10)
