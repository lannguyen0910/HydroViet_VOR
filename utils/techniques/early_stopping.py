from utils.helper.saved_model import save_model
import numpy as np


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimizer, scheduler, fold, epoch):
        if self.val_loss_min == np.Inf:
            self.val_loss_min = val_loss

        elif val_loss > self.val_loss_min:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(
                self.counter, self.patience))

            if self.counter >= self.patience:
                print('Early Stopping - Fold {} Training is Stopping'.format(fold))
                self.early_stop = True

        else:  # val_loss < val_loss_min
            save_model(model, optimizer, scheduler, fold, epoch, best=True)
            print('*** Validation loss decreased ({} --> {}).  Saving model... ***'.
                  format(round(self.val_loss_min, 6), round(val_loss, 6)))

            self.val_loss_min = val_loss
            self.counter = 0
