from models.classifier import Classifier
from random import shuffle
from numpy.lib.npyio import save
from torchvision.models.resnet import ResNet,  resnet34, resnet50
from tqdm import tqdm
from datasets.image_classification import ImageClassificationDataset
from utils.getter import *


# from torchsummary import summary


img_size = (300, 300)
transforms = Compose([
    Resize(img_size),
    ToTensor(),
    Normalize()
])

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# normalize = transforms.Normalize(mean=mean, std=std)
# transform_train = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ])

# transform_val = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.ToTensor(),
#     normalize
# ])


if __name__ == '__main__':
    train_set = ImageClassificationDataset(
        root='trainingSet/trainingSet', transforms=transforms, shuffle=True)
    val_set = ImageClassificationDataset(
        root='trainingSample', transforms=transforms,  shuffle=False)
    print('Train set: ', train_set)
    print('Val set: ', val_set)
    print('Item: ', train_set[2]['category'])
    N_CATEGORIES = len(train_set)

    logger = Logger()
    logger.log('Tensorboard for Image Classification')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    BATCH_SIZE = 2
    train_loader = data.DataLoader(
        train_set, batch_size=BATCH_SIZE, collate_fn=train_set.collate_fn, shuffle=True)
    val_loader = data.DataLoader(
        val_set, batch_size=BATCH_SIZE, collate_fn=val_set.collate_fn, shuffle=False)
    # trainimages, trainlabels = next(iter(train_loader))
    # print('Train images: ', trainimages)
    # print('Train_label: ', trainlabels)

    EPOCHS = 1000
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    metrics = [ClassificationAccuracyMetric(decimals=3)]
    optimizer = torch.optim.Adam

    model = Classifier(n_classes=N_CATEGORIES, optimizer=optimizer, criterion=criterion, metrics=metrics,
                       lr=lr, freeze=True, device=device, optim_params=None)
    print('Number of trainable parameters in model: ',
          model.trainable_parameters())

    chpoint = CheckPoint(save_per_epoch=2)
    scheduler = StepLR(model.optimizer, step_size=2, gamma=0.1)

    trainer = Trainer(model, train_loader, val_loader,
                      checkpoint=chpoint, scheduler=scheduler, evaluate_epoch=2)
    trainer.fit(num_epochs=EPOCHS, print_per_iter=100)
