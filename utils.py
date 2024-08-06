import torch
import numpy as np
import gc
import os

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return


"""Loading in Models from Model Library"""
def extract_epoch_name(filename):
    name_part = os.path.splitext(filename)[0]  # Remove the file extension
    if '-' in name_part:
        numerator, denominator = map(int, name_part.split('-'))
        return numerator / denominator
    else:
        return int(name_part)

def get_sorted_epoch_names(dirname):
    all_files = os.listdir(dirname)
    epochs = [os.path.splitext(file)[0] for file in all_files]
    epoch_list = sorted(epochs, key=extract_epoch_name)

    return epoch_list