import torch
import numpy as np
import gc

class AverageMeter(object):
    """ Computes and stores the average and current value. """
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val
        self.count += n

    def avg(self):
        return self.sum / self.count

def set_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

device = set_torch_device()

def set_seed(SEED):
    ###### set the seed
    #random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    ######
    print(f'set seed to {SEED}')
    return

def clear_memory():
    gc.collect()
    with torch.no_grad():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    return

def check_memory():
    total = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    check = f'CUDA currently has {(allocated/total)*100:.2f}% memory allocated, {(reserved/total)*100:.2f}% memory reserved'
    print(check)
    return