from random import shuffle
from numpy.lib.npyio import save
from torchvision.models.resnet import ResNet,  resnet34
from tqdm import tqdm
from datasets.transform import transforming
from datasets.image_classification import ImageClassificationDataset
from utils.getter import *
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


if __name__ == '__main__':
    train_set = ImageClassificationDataset(root='trainingSet',
                                           transforms=transforming(
                                               img_size=(224, 224)), n_samples=100, shuffle=True)
    val_set = ImageClassificationDataset(root='trainingSample',
                                         transforms=transforming(
                                             img_size=(224, 224)), n_samples=10, shuffle=True)
    # print(train_set)
    # print(val_set)
    N_CATEGORIES = len(train_set.n_classes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    BATCH_SIZE = 2
    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    EPOCHS = 10
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    metrics = [ClassificationF1Score(
        N_CATEGORIES, average='macro'), ClassificationAccuracyMetric(decimals=3)]
    model = resnet34()
    summary(model, (3, 224, 224))
    # model = BaselineModel(lr=lr, criterion=criterion, optimizer=optimizer,
    #                       metrics=metrics, device=device)

    chpoint = CheckPoint(save_per_epoch=2)

    trainer = Trainer(model, train_loader, val_loader,
                      checkpoint=chpoint, evaluate_per_epoch=2)
    trainer.fit(num_epochs=EPOCHS, print_per_iter=10)
