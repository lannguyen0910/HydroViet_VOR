import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckPoint, checkpoint


class Trainer(nn.Module):
    def __init__(self, model, train_loader, val_loader, checkpoint=None, evaluate_per_epoch=1):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.metrics = model.metrics  # list
        self.checkpoint = checkpoint
        self.evaluate_per_epoch = evaluate_per_epoch

    def fit(self, num_epochs=10, print_per_iter=None):
        self.num_epochs = num_epochs
        self.num_iters = num_epochs * len(self.train_loader)

        if self.checkpoint is None:
            self.checkpoint = CheckPoint(save_per_epoch=int(num_epochs/10) + 1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.train_loader) / 10)

        print(f'Training for {num_epochs} ...')
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss = self.train_per_epoch()
            print(f'Epoch: [{epoch + 1}|{num_epochs}]')

            if epoch % self.evaluate_per_epoch == 0 and epoch + 1 >= self.evaluate_per_epoch:
                val_loss, val_metrics = self.evaluation_per_epoch()
                print(f'Eval: Val Loss: {val_loss:10.5f} |', end=' ')
                for metric, score in val_metrics.items():
                    print(f'{metric}: {score}', end='|')
                print('\n')

            if epoch % self.checkpoint.save_per_epoch == 0 and epoch + 1 == num_epochs:
                self.checkpoint.save(self.model, epoch=epoch)

    def train_per_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            loss = self.model.training_step(batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()

            if i % self.print_per_iter == 0 or i == len(self.train_loader) - 1 and i != 0:
                print(f'\tIter: [{len(self.train_loader)*self.epoch + i + 1}/{self.num_iters}] \
                    | Traning Loss: {running_loss/self.print_per_iter:10.5f}')
