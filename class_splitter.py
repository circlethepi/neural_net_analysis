import torch
import copy
from torchvision import datasets
import torch.utils.data.dataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

import neural_network as nn_mod

def set_seed(SEED):
    ###### set the seed
    #random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    ######
    print(f'set seed to {SEED}')
    return

#set_seed()

def single_class(dataset, single_class):
    for i in range(len(dataset.targets)):
        if dataset.targets[i] != single_class:
            dataset.targets[i] = -1
        else:
            dataset.targets[i] = 1

    for j in range(len(dataset.targets)):
        if dataset.targets[j] == -1:
            dataset.targets[j] = 0


def make_all_datasets(dataset_class=datasets.CIFAR10):
    whole_train, whole_val = nn_mod.get_datasets(dataset_class=dataset_class)

    datasets = {}
    for j in range(len(whole_train.classes)):
        train, val = nn_mod.get_datasets(dataset_class=dataset_class)
        single_class(train, j)
        single_class(val, j)
        datasets[j] = (copy.deepcopy(train), copy.deepcopy(val))

    print(f'New dataset created for each ')
    return datasets


def make_all_single_loaders(batch_size=64, dataset_class=datasets.CIFAR10):
    datasets = make_all_datasets(dataset_class=dataset_class)
    dataloaders = {}
    for c in datasets.keys():
        train_load = nn_mod.get_dataloader(batch_size, datasets[c][0], shuffle=True)
        val_load = nn_mod.get_dataloader(batch_size, datasets[c][1], shuffle=False)

        dataloaders[c] = (train_load, val_load)

    return dataloaders

### The following is better implemented in perturbation.py
def subset_class_loader(class_indices, batch_size=64, dataset_class=datasets.CIFAR10, mod_ind=None, columns=None,
                        rows=None, val=255, intensity=False):
    #if pad_level > 4 or pad_level < 0 or type(pad_level) != int:
    #    raise Exception('Please enter valid integer pad level between 0 and 3')

    print(class_indices)#, type(class_indices))
    # get unmodified classes first
    # load the entire dataset
    trainset, valset = nn_mod.get_datasets(dataset_class=dataset_class)

    # get just the subset
    indices_train = [i for i, (e, c) in enumerate(trainset) if c in class_indices]
    indices_val = [i for i, (e, c) in enumerate(valset) if c in class_indices]

    # get the subset
    trainset_sub = torch.utils.data.Subset(trainset, indices_train)
    valset_sub = torch.utils.data.Subset(valset, indices_val)

    # setting up the mod transform if it exists
    #mod_trans = transforms.Compose(transforms.Pad(padding=8))

    # if mod class, then mod class
    if mod_ind:
        # get the indices
        mind_train = [i for i, (e, c) in enumerate(trainset) if c in mod_ind]
        mind_val = [i for i, (e, c) in enumerate(valset) if c in mod_ind]

        # get the subsets
        m_train_sub = torch.utils.data.Subset(trainset, mind_train)
        m_val_sub = torch.utils.data.Subset(valset, mind_val)

        #pad = (2, 4, 8, 16, 32)[pad_level]
        #pad = pad_level
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mod_transform = transforms.Compose([#transforms.Pad(padding=pad),
                                            #transforms.ToTensor(),
                                            #transforms.Resize(size=32),
                                            transforms.Normalize(mean=mean, std=std),
                                            colrow_colors(column_indices=columns, row_indices=rows,
                                                          val=val, intensity=intensity)
        ])

        modded_train = MyDataset(m_train_sub, transform=mod_transform)
        modded_val = MyDataset(m_val_sub, transform=mod_transform)

        trainset_sub = torch.utils.data.ConcatDataset([trainset_sub, modded_train])
        valset_sub = torch.utils.data.ConcatDataset([valset_sub, modded_val])


    # get the dataloader
    train_loader_sub = nn_mod.get_dataloader(batch_size, trainset_sub, shuffle=True)
    val_loader_sub = nn_mod.get_dataloader(batch_size, valset_sub, shuffle=False)

    return train_loader_sub, val_loader_sub

###################################
## Dataset and Transform Classes ##
###################################

# custom dataset to apply transforms to
class MyDataset:
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# custom transform for columns
class colrow_colors(object):
    def __init__(self, val=255, column_indices=None, row_indices=None, intensity=False):
        self.val = val
        self.columns = column_indices
        self.rows = row_indices
        self.intensity = intensity
    def __call__(self, img_tensor):  # this should always come after the toTensor transform!
        """

        :param img_tensor: image in the form of a tensor
        :param val:
        :param column_indices:
        :param row_indices:
        :return:
        """
        # Do some transformations. Here, we're just passing though the input
        if self.columns:
            for index in self.columns:
                img_tensor[:, :, index] = torch.tensor(self.val)

        if self.rows:
            for index in self.rows:
                img_tensor[:, index, :] = torch.tensor(self.val)

        if self.intensity:
            img_tensor = torch.clamp(img_tensor, min=0, max=self.val)

        return img_tensor

    def __repr__(self):
        return "Turning Columns and or Rows a Color"
