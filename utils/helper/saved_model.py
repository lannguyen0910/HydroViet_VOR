import torch
import os


# Saved model with KFold
def save_model(model, optimizer, scheduler, fold, epoch, save_every=False, best=False):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    if save_every == True:
        if not (os.path.isdir('./saved_model')):
            os.mkdir('./saved_model')
        torch.save(
            state, './saved_model/fold_{}_epoch_{}'.format(fold+1, epoch+1))
    if best == True:
        if not (os.path.isdir('./best_model')):
            os.mkdir('./best_model')
        torch.save(
            state, './best_model/fold_{}_epoch_{}'.format(fold+1, epoch+1))
