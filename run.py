from random import shuffle
from tqdm import tqdm
from datasets.transform import transforming
from datasets.image_classification import ImageClassificationDataset
from utils.getter import *
import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision.models as models


if __name__ == '__main__':
    train_set = ImageClassificationDataset(root='trainingSet',
                                           transforms=transforming(
                                               img_size=(224, 224)), n_samples=100, shuffle=True)
    val_set = ImageClassificationDataset(root='trainingSample',
                                         transforms=transforming(
                                             img_size=(224, 224)), n_samples=10, shuffle=True)
    # print(train_set)
    # print(val_set)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    BATCH_SIZE = 2
    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)

    EPOCHS = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
