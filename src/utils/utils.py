import random
import numpy as np
import torch
import os

def set_rand_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONNASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True