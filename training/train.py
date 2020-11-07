import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckPoint, load


class Trainer(nn.Module):
    def __init__(self, model, train_loader, val_loader, **kwargs):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.metrics = model.metrics  # list
        self.set_attribute(kwargs)

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
        for epoch in tqdm(range(num_epochs)):
            self.epoch = epoch
            train_loss = self.train_per_epoch()
            print(
                f'Epoch: [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss}')

            if epoch % self.evaluate_epoch == 0 and epoch + 1 >= self.evaluate_epoch:
                val_loss, val_acc, val_metrics = self.evaluate_per_epoch()
                print(
                    f'Eval: Val Loss: {val_loss:10.5f} | Val Acc: {val_acc:10.5f} |', end=' ')
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

            iters = len(self.train_loader)*self.epoch + i + 1
            if i % self.print_per_iter == 0:
                print(f'\tIter: [{iters}/{self.num_iters}] \
                    | Traning Loss: {running_loss/self.print_per_iter:10.5f}')
                running_loss = 0

        return epoch_loss / len(self.train_loader)

    def inference_per_batch(self, test_loader):
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = self.model.inference_step(batch)
                for i in outputs:
                    results.append(i)

        return results

    def evaluate_per_epoch(self):
        self.model.eval()
        epoch_loss = 0.0
        epoch_acc = 0.0
        metric_dict = {}

        with torch.no_grad():
            for batch in self.val_loader:
                loss, accuracy, metrics = self.model.evaluate_step(batch)
                epoch_loss += loss
                epoch_acc += accuracy
                metric_dict.update(metrics)

        self.model.reset_metrics()
        return epoch_loss / len(self.val_loader), epoch_acc / len(self.val_loader), metric_dict

    def __str__(self) -> str:
        title = '------------- Model Summary ---------------\n'
        name = f'Name: {self.model.name}\n'
        params = f'Number of params: {self.model.trainable_parameters}\n'
        loss = f'Loss function: {self.criterion[:-2]} \n'
        train_iter_per_epoch = f'Number of train iterations per epoch: {len(self.train_loader)}\n'
        val_iter_per_epoch = f'Number of val iterations per epoch: {len(self.val_loader)}'

        return title + name + params + loss + train_iter_per_epoch + val_iter_per_epoch

    def print_forward_step(self):
        self.model.eval()
        outputs = self.model.forward_step()
        print('Feedforward: output_shape: ', outputs.shape)

    def set_attribute(self, kwargs):
        self.checkpoint = None
        self.evaluate_epoch = 1
        for i, j in kwargs.items():
            setattr(self, i, j)
