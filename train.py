from datasets.object_retrieval.datasethelper import collate_fn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from functools import partial
from utils.getter import *
# from datasets.image_classification import ImageClassificationDataset
# from tqdm import tqdm
# from torchvision.models.resnet import ResNet,  resnet34, resnet50
# from numpy.lib.npyio import save
# from random import shuffle
# from models.classifier import Classifier
import argparse
import pandas as pd
import shutil
from torchvision import transforms
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


def main():
    set_seed()
    global best_acc, plotter
    plotter = VisdomLinePlotter(env_name='VOR')

    kwargs = {'num_workers': hp.num_workers,
              'pin_memory': hp.pin_memory} if use_gpu else {}

    train_df = pd.read_csv(hp.train_csv)
    val_df = pd.read_csv(hp.val_csv)

    q_train = TripletDataset(
        root=hp.image_dir, df=train_df,  shuffle=True, mode='train')
    # q_train = QueryExtractor(dataset='paris', image_dir='./dataset/paris/image/', label_dir='./dataset/paris/label/', subset='train')

    train_dataloader = DataLoader(q_train, batch_size=1, num_workers=1, drop_last=False,
                                  shuffle=True, collate_fn=partial(collate_fn, augment=True), pin_memory=False)

    trainset = TripletDataset(
        root='train', df=train_df, transform=transforms_train, shuffle=True, mode='train')
    trainloader = DataLoader(
        trainset, batch_size=hp.batch_size, shuffle=True, **kwargs)

    valset = TripletDataset(
        root='testing', df=val_df, transform=transforms_val, shuffle=True, mode='val')
    valloader = DataLoader(
        valset, batch_size=hp.batch_size, shuffle=True, **kwargs)

    model = TripletNet(ResNetExtractor(version=101))
    print('Number of parameters: ', count_parameters(model))

    model.apply(weights_init)
    model = torch.jit.script(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    criterion = torch.jit.script(
        torch.nn.MarginRankingLoss(margin=0.2)).to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(1, hp.epochs + 1):
        train(trainloader, model, criterion, optimizer, scheduler, epoch)

        acc = test(valloader, model, criterion, epoch)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_pred': best_acc,
        }, is_best)


def train(train_loader, tnet, criterion, optimizer, scheduler, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if use_gpu:
            data1, data2, data3 = Variable(data1).to(device), Variable(
                data2).to(device), Variable(data3).to(device)

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(
            data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        # print(data1.shape)
        # print(dista.shape)
        # print(target.shape)
        # print(data1.shape)

        if use_gpu:
            target = Variable(target).to(device)

        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(
            2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # # measure accuracy and record loss
        # acc = accuracy(dista, distb)

        losses.update(loss_triplet.data.cpu().numpy(), data1.size(0))
        # accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % hp.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  #   'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                      epoch, batch_idx * len(data1), len(train_loader.dataset),
                      losses.val, losses.avg,
                      #   100. * accs.val, 100. * accs.avg,
                      emb_norms.val, emb_norms.avg))

    scheduler.step(losses.avg)
    # log avg values to somewhere
    # plotter.plot('acc', 'train', 'class acc', epoch, accs.avg)
    plotter.plot('loss', 'train', 'class loss', epoch, losses.avg)
    plotter.plot('emb_norms', 'train', 'class emb', epoch, emb_norms.avg)


def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    with torch.no_grad():
        for _, (data1, data2, data3) in enumerate(test_loader):
            if use_gpu:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
            data1, data2, data3 = Variable(
                data1), Variable(data2), Variable(data3)

            accuracies = [0, 0, 0]
            acc_threshes = [0, 0.2, 0.5]
            # compute output
            dista, distb, _, _, _ = tnet(data1, data2, data3)
            target = torch.FloatTensor(dista.size()).fill_(1)
            target = Variable(target).to(device)

            test_loss = criterion(dista, distb, target).data.cpu().numpy()

            # measure accuracy and record loss
            for i in range(len(accuracies)):
                prediction = (distb - dista - hp.margin *
                              acc_threshes[i]).cpu().data
                prediction = prediction.view(prediction.numel())
                prediction = (prediction > 0).float()
                batch_acc = prediction.sum() * 1.0 / prediction.numel()
                accuracies[i] += batch_acc

            # accs.update(acc, data1.size(0))
            losses.update(test_loss, data1.size(0))

    print('\nTest set: Average loss: {:.4f}'.format(losses.avg))
    for i in range(len(accuracies)):
        print('Test Accuracy with diff = {}% of margin: {}'.format(
            acc_threshes[i] * 100, accuracies[i] / len(test_loader)))
    print("****************************************************************\n")

    # plotter.plot('acc', 'test', 'class acc', epoch, accs.avg)
    plotter.plot('loss', 'test', 'class loss', epoch, losses.avg)
    return accs.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (hp.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' %
                        (hp.name) + 'model_best.pth.tar')


if __name__ == '__main__':
    main()
