import torch
import numpy as np
import random
import os

SEED = 42


def set_seed(seed=SEED, device=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    if device == True:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        torch.manual_seed(seed)
