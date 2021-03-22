from utils.techniques.gradient_clipping import clip_gradient
import torch
import torch.nn as nn
from tqdm import tqdm
from .checkpoint import CheckPoint, load
from logger import Logger
import time
import os
from augmentation import Denormalize
import cv2
import numpy as np
from utils.gradcam import *


class Trainer(nn.Module):
    def __init__(self, config, model, train_loader, val_loader, **kwargs):
        super().__init__()
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.metrics = model.metrics  # list of classification metrics
        self.set_attribute(kwargs)

    def logged(self, logs):
        tags = [tag for tag in logs.keys()]
        values = [value for value in logs.values()]

        self.logger.write(tags=tags, values=values)

    def fit(self,  start_epoch=0, start_iter=0, num_epochs=10, print_per_iter=None):
        self.num_epochs = num_epochs
        self.num_iters = num_epochs * len(self.train_loader)

        if self.checkpoint is None:
            self.checkpoint = CheckPoint(save_per_epoch=int(num_epochs/10) + 1)

        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.train_loader) / 10)

        self.epoch = start_epoch

        print(f'===========================START TRAINING=================================')
        print(f'Training for {num_epochs} epochs ...')
        for epoch in range(self.epoch, self.num_epochs):
            try:
                self.epoch = epoch
                self.train_per_epoch()

                if self.num_evaluate_per_epoch != 0:
                    if epoch % self.num_evaluate_per_epoch == 0 and epoch+1 >= self.num_evaluate_per_epoch:
                        self.evaluate_per_epoch()

                if self.scheduler is not None and self.step_per_epoch:
                    self.scheduler.step()
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Learning rate/Epoch': lr}
                    self.logged(log_dict)

            except KeyboardInterrupt:
                self.checkpoint.save(self.model, save_mode='last', epoch=self.epoch,
                                     iters=self.iters, best_value=self.best_value)
                print("Stop training, checkpoint saved...")
                break

        print("Training Completed!")

    def train_per_epoch(self):
        self.model.train()

        running_loss = 0.0
        running_time = 0

        loop = tqdm(self.train_loader)
        for i, batch in enumerate(loop):
            start_time = time.time()

            with torch.cuda.amp.autocast():
                loss, loss_dict = self.model.training_step(batch)
                if self.use_accumulate:
                    loss /= self.accumulate_steps

            self.model.scaler(loss, self.optimizer)

            if self.use_accumulate:
                if (i+1) % self.accumulate_steps == 0 or i == len(self.train_loader)-1:
                    self.model.scaler.step(
                        self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                    self.optimizer.zero_grad()

                    if self.scheduler is not None and not self.step_per_epoch:
                        self.scheduler.step(
                            (self.num_epochs + i) / len(self.train_loader))
                        lrl = [x['lr'] for x in self.optimizer.param_groups]
                        lr = sum(lrl) / len(lrl)
                        log_dict = {'Learning rate/Iterations': lr}
                        self.logging(log_dict)
            else:
                self.model.scaler.step(
                    self.optimizer, clip_grad=self.clip_grad, parameters=self.model.parameters())
                self.optimizer.zero_grad()
                if self.scheduler is not None and not self.step_per_epoch:
                    # self.scheduler.step()
                    self.scheduler.step(
                        (self.num_epochs + i) / len(self.train_loader))
                    lrl = [x['lr'] for x in self.optimizer.param_groups]
                    lr = sum(lrl) / len(lrl)
                    log_dict = {'Learning rate/Iterations': lr}
                    self.logging(log_dict)

            torch.cuda.synchronize()

            end_time = time.time()

            for (key, value) in loss_dict.items():
                if key in running_loss.keys():
                    running_loss[key] += value
                else:
                    running_loss[key] = value

            running_time += end_time-start_time
            self.iters = self.start_iter + \
                len(self.train_loader)*self.epoch + i + 1
            if self.iters % self.print_per_iter == 0:

                for key in running_loss.keys():
                    running_loss[key] /= self.print_per_iter
                    running_loss[key] = np.round(running_loss[key], 5)
                loss_string = '{}'.format(running_loss)[
                    1:-1].replace("'", '').replace(",", ' ||')
                print("[{}|{}] [{}|{}] || {} || Time: {:10.4f}s".format(
                    self.epoch, self.num_epochs, self.iters, self.num_iters, loss_string, running_time))
                self.logging(
                    {"Training Loss/Batch": running_loss['T'] / self.print_per_iter, })
                running_loss = {}
                running_time = 0

            if (self.iters % self.checkpoint.save_per_iter == 0 or self.iters == self.num_iters - 1):
                print(f'Save model at [{self.epoch}|{self.iters}] to last.pth')
                self.checkpoint.save(
                    self.model,
                    save_mode='last',
                    epoch=self.epoch,
                    iters=self.iters,
                    best_value=self.best_value)

    def evaluate_per_epoch(self):
        self.model.eval()
        epoch_loss = {}
        metric_dict = {}

        print('=============================EVALUATION===================================')
        start_time = time.time()
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                _, loss_dict = self.model.evaluate_step(batch)

                for (key, val) in loss_dict.items():
                    if key in epoch_loss.keys():
                        epoch_loss[key] += val
                    else:
                        epoch_loss[key] = val

        end_time = time.time()
        running_time = end_time - start_time
        metric_dict = self.model.get_metric_values()

        self.model.reset_metrics()

        for key in epoch_loss.keys():
            epoch_loss[key] /= len(self.val_loader)
            epoch_loss[key] = np.round(epoch_loss[key], 5)
        loss_string = '{}'.format(epoch_loss)[
            1:-1].replace("'", '').replace(",", ' ||')
        print()
        print("[{}|{}] || {} || Time: {:10.4f} s".format(
            self.epoch, self.num_epochs, loss_string, running_time))

        for metric, score in metric_dict.items():
            print(metric + ': ' + str(score), end=' | ')
        print()
        print('==========================================================================')

        log_dict = {
            "Validation Loss/Epoch": epoch_loss['T'] / len(self.val_loader), }
        log_dict.update(metric_dict)
        self.logging(log_dict)

        # Save model gives best mAP score
        if metric_dict['acc'] > self.best_value:
            self.best_value = metric_dict['acc']
            self.checkpoint.save(self.model, save_mode='best', epoch=self.epoch,
                                 iters=self.iters, best_value=self.best_value)

        if self.visualize_when_val:
            self.visualize_batch()

    def visualize_batch(self):
        # Vizualize Grad Class Activation Mapping
        if not os.path.exists('./samples'):
            os.mkdir('./samples')

        denom = Denormalize()
        batch = next(iter(self.val_loader))
        images = batch["imgs"]
        #targets = batch["targets"]

        self.model.eval()

        config_name = self.cfg.model_name.split('_')[0]
        grad_cam = GradCam(model=self.model.model, config_name=config_name)

        for idx, inputs in enumerate(images):
            image_outname = os.path.join(
                'samples', f'{self.epoch}_{self.iters}_{idx}.jpg')
            img_show = denom(inputs)
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(self.model.device)
            target_category = None
            grayscale_cam, label_idx = grad_cam(inputs, target_category)
            label = self.cfg.obj_list[label_idx]
            img_cam = show_cam_on_image(img_show, grayscale_cam, label)
            cv2.imwrite(image_outname, img_cam)

    def __str__(self) -> str:
        title = '------------- Model Summary ---------------\n'
        name = f'Name: {self.model.name}\n'
        params = f'Number of params: {self.model.trainable_parameters}\n'

        train_iter_per_epoch = f'Number of train iterations per epoch: {len(self.train_loader)}\n'
        val_iter_per_epoch = f'Number of val iterations per epoch: {len(self.val_loader)}'

        return title + name + params + train_iter_per_epoch + val_iter_per_epoch

    def print_forward_step(self):
        self.model.eval()
        outputs = self.model.forward_step()
        print('Feedforward: output_shape: ', outputs.shape)

    def set_accumulate_step(self):
        self.use_accumulate = False
        if self.config.total_accumulate_steps > 0:
            self.use_accumulate = True
            self.accumulate_steps = max(
                round(self.config.total_accumulate_steps / self.config.batch_size), 1)

    def set_amp(self):
        self.use_amp = False
        if self.config.mixed_precision:
            self.use_amp = True

    def set_attribute(self, **kwargs):
        self.checkpoint = None
        self.evaluate_epoch = 1
        self.scheduler = None
        self.gradient_clip = 10
        self.visualize_when_val = True
        self.step_per_epoch = False
        self.num_evaluate_per_epoch = 1
        self.best_value = 0.0
        self.logger = Logger()
        self.set_accumulate_step()
        self.set_amp()

        for i, j in kwargs.items():
            setattr(self, i, j)
