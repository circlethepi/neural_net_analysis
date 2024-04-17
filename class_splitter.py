import torch
import copy
from torchvision import datasets

import neural_network as nn_mod


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

def subset_class_loader(class_indices, batch_size=64, dataset_class=datasets.CIFAR10):
    trainset, valset = nn_mod.get_datasets(dataset_class=dataset_class)
    indices_train = [i for i, (e, c) in enumerate(trainset) if c in class_indices]
    indices_val = [i for i, (e, c) in enumerate(valset) if c in class_indices]

    # get the subset
    trainset_sub = torch.utils.data.Subset(trainset, indices_train)
    valset_sub = torch.utils.data.Subset(valset, indices_val)

    # get the dataloader
    train_loader_sub = nn_mod.get_dataloader(batch_size, trainset_sub, shuffle=True)
    val_loader_sub = nn_mod.get_dataloader(batch_size, valset_sub, shuffle=False)

    return train_loader_sub, val_loader_sub

