import torch

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
        idstr = 'cuda'
    elif torch.backends.mps.is_available():
        idstr = 'mps'
    else:
        idstr = 'cpu'
    
    device = torch.device(idstr)
    return device

device = set_torch_device()
