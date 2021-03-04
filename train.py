import argparse
from visdom import Visdom
import pandas as pd

from models.classifier import Classifier
from random import shuffle
from numpy.lib.npyio import save
from torchvision.models.resnet import ResNet,  resnet34, resnet50
from tqdm import tqdm
from datasets.image_classification import ImageClassificationDataset
from utils.getter import *
from torch.utils.data import DataLoader

# from torchsummary import summary


# img_size = (300, 300)
# transforms = Compose([
#     Resize(img_size),
#     ToTensor(),
#     Normalize()
# ])

# # mean = [0.485, 0.456, 0.406]
# # std = [0.229, 0.224, 0.225]
# # normalize = transforms.Normalize(mean=mean, std=std)
# # transform_train = transforms.Compose([
# #     transforms.Resize(img_size),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),
# #     normalize
# # ])

# # transform_val = transforms.Compose([
# #     transforms.Resize(img_size),
# #     transforms.ToTensor(),
# #     normalize
# # ])


# if __name__ == '__main__':
#     train_set = ImageClassificationDataset(
#         root='trainingSet/trainingSet', transforms=transforms, shuffle=True)
#     val_set = ImageClassificationDataset(
#         root='trainingSample', transforms=transforms,  shuffle=False)
#     print('Train set: ', train_set)
#     print('Val set: ', val_set)
#     print('Item: ', train_set[2]['category'])
#     N_CATEGORIES = len(train_set)

#     logger = Logger()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(device)

#     BATCH_SIZE = 2
#     train_loader = data.DataLoader(
#         train_set, batch_size=BATCH_SIZE, collate_fn=train_set.collate_fn, shuffle=True)
#     val_loader = data.DataLoader(
#         val_set, batch_size=BATCH_SIZE, collate_fn=val_set.collate_fn, shuffle=False)
#     # trainimages, trainlabels = next(iter(train_loader))
#     # print('Train images: ', trainimages)
#     # print('Train_label: ', trainlabels)

#     EPOCHS = 1000
#     lr = 1e-3
#     criterion = nn.CrossEntropyLoss()
#     metrics = [ClassificationAccuracyMetric(decimals=3)]
#     optimizer = torch.optim.Adam

#     model = Classifier(n_classes=N_CATEGORIES, optimizer=optimizer, criterion=criterion, metrics=metrics,
#                        lr=lr, freeze=True, device=device, optim_params=None)
#     print('Number of trainable parameters in model: ',
#           model.trainable_parameters())

#     chpoint = CheckPoint(save_per_epoch=2)
#     scheduler = StepLR(model.optimizer, step_size=2, gamma=0.1)

#     trainer = Trainer(model, train_loader, val_loader,
#                       checkpoint=chpoint, scheduler=scheduler, evaluate_epoch=2)
#     trainer.fit(num_epochs=EPOCHS, print_per_iter=100)
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')
best_acc = 0

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
transform_train = transforms.Compose([
    transforms.Resize(hp.size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_val = transforms.Compose([
    transforms.Resize(hp.size),
    transforms.ToTensor(),
    normalize
])


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array(
                [y]), env=self.env, win=self.plots[var_name], name=split_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]


def main():
    set_seed()
    global best_acc, plotter
    plotter = VisdomLinePlotter(env='VOR')

    kwargs = {'num_workers': hp.num_workers,
              'pin_memory': hp.pin_memory} if use_gpu else {}

    train_df = pd.read_csv('write.csv')
    trainset = TripletDataset(
        root='train', df=train_df, transforms=transform_train, shuffle=True, mode='train')
    trainloader = DataLoader(
        trainset, batch_size=hp.batch_size, shuffle=True, **kwargs)

    testset = TripletDataset(
        root='testing', transforms=transform_val, shuffle=True, mode='test')
    testloader = DataLoader(
        testset, batch_size=hp.batch_size, shuffle=True, **kwargs)

    model = TripletNet(ResNetExtractor(version=101))
    model.apply(weights_init)
    model = torch.jit.script(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    criterion = torch.jit.script(TripletLoss())

    
