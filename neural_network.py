# imports
# interactivity
from tqdm import tqdm
#from tqdm.notebook import tqdm
from ipywidgets import FloatProgress
import time
import copy
from utils import AverageMeter

# torch things
import torch
from torch import nn
import torch.utils.data.dataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets

# basics
import numpy as np
from matplotlib import pyplot as plt
import random

# other packages/files

##################################
# Setting up the Neural Networks #
##################################

# Linear Neural Network
class Neural_Network(nn.Module):
    def __init__(self, n_neurons: list, num_classes=10, input_size=32*32*3):
        print(f"Initializing {len(n_neurons)} layer model")
        super(Neural_Network, self).__init__()

        # creating the model
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(nn.Linear(input_size, n_neurons[0]))
        # the hidden layers
        for k in range(len(n_neurons)-1):
            self.layers.append(nn.Linear(n_neurons[k], n_neurons[k+1]))
        # output layer
        self.layers.append(nn.Linear(n_neurons[-1], num_classes))

        # setting some attributes
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.input_size = input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        i = 0
        for layer in self.layers:
            x = layer(x)
            if i < len(self.n_neurons) - 2:
                x = torch.relu(x)
        i += 1
        return x


# Convolutional Neural Network (based on VGG)



#######################
## Dataset Functions ##
#######################

def get_datasets(dataset_class=datasets.CIFAR10):
    """

    :param dataset_class: the class of the dataset to use
    :return: datasets to use in the
    """
    #mean = [0.485, 0.456, 0.406]
    #std = [0.229, 0.224, 0.225]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    if dataset_class != datasets.MNIST:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[mean[0]], std=[std[0]])]
        )

    def get_dataset(train: bool):  # Returns the train or validation dataset.
        root = "./data"
        kwargs = dict(root=root, transform=transform, train=train, download=True)
        dataset = dataset_class(**kwargs)
        return dataset

    train_dataset = get_dataset(train=True)
    val_dataset = get_dataset(train=False)
    return train_dataset, val_dataset


def get_dataloader(batch_size, dataset, shuffle):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=0, pin_memory=True)


def get_dataloaders(batch_size=64, dataset_class=datasets.CIFAR10):
    """
    Returns train and validation dataloaders.

    :param batch_size:
    :param dataset_class:
    :return: dataloaders
    """
    train_dataset, val_dataset = get_datasets(dataset_class=dataset_class)

    train_loader = get_dataloader(batch_size, train_dataset, shuffle=True)
    val_loader = get_dataloader(batch_size, val_dataset, shuffle=False)
    return train_loader, val_loader


############################
# Split Dataset Generation #
"""USE PERTURBATION IMPLEMENTATION INSTEAD"""
############################
# create dictionary with the classes and the number
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_dict = dict(zip(classes, list(range(10))))


def create_training_subset(whole_trainset, classlist):
    # convert the classnames into the class indices
    classnums = []
    for c in classlist:
        classnums.append(class_dict[c])

    # get the indices
    index_list = [i for i, (e, c) in enumerate(whole_trainset) if c in classnums]

    # get the split set
    split = torch.utils.data.Subset(whole_trainset, index_list)

    return split


def create_class_set_splits(whole_trainset, classlists):
    sets = [create_training_subset(whole_trainset, li) for li in classlists]
    return sets


def get_class_set_split_dataloaders(batch_size, class_lists, dataset_class=datasets.CIFAR10):
    # get train and val
    trainset, valset = get_datasets(dataset_class=dataset_class)

    # get the train sets
    dsets = create_class_set_splits(trainset, class_lists)

    # get the train loaders
    loaders = []
    for ds in dsets:
        load = get_dataloader(batch_size, ds, shuffle=True)
        loaders.append(load)

    # get the val loader
    val_load = get_dataloader(batch_size, valset, shuffle=False)

    return loaders, val_load


def get_split_class_dataloaders(batch_size, dataset_class=datasets.CIFAR10, n_classes=5, random_classes=False):
    train_dataset, val_dataset = get_datasets(dataset_class=dataset_class)

    classnums = list(range(10))
    if random_classes:
        random.shuffle(classnums)
    print(f'Class set 1: {classnums[0:n_classes]}\nClas set 2: {classnums[n_classes+1:]}')

    # get the indices for the trainsets
    first_half_indices = [i for i, (e, c) in enumerate(train_dataset) if c in classnums[:n_classes]]
    second_half_indices = [j for j, (e, c) in enumerate(train_dataset) if c in classnums[n_classes+1:]]

    # get the split trainsets
    trainset_1 = torch.utils.data.Subset(train_dataset, first_half_indices)
    trainset_2 = torch.utils.data.Subset(train_dataset, second_half_indices)

    # get the dataloaders
    train_loader_1 = get_dataloader(batch_size, trainset_1, shuffle=True)
    train_loader_2 = get_dataloader(batch_size, trainset_2, shuffle=True)
    val_loader = get_dataloader(batch_size, val_dataset, shuffle=False)

    return train_loader_1, train_loader_2, val_loader


def split_in_half_loaders(batch_size, dataset_class=datasets.CIFAR10):
    trainset, valset = get_datasets(dataset_class=dataset_class)

    print(len(trainset))

    train1, train2 = random_split(trainset, [int(len(trainset)/2), int(len(trainset)/2)])

    # get the dataloaders
    train_loader_1 = get_dataloader(batch_size, train1, shuffle=True)
    train_loader_2 = get_dataloader(batch_size, train2, shuffle=True)
    val_loader = get_dataloader(batch_size, valset, shuffle=False)

    return train_loader_1, train_loader_2, val_loader


#######################
# Variance Adjustment #
#######################
# creates a weights initialization function where you can adjust the variance
def create_weights_function(vary):
    """
    initializes a network with a gaussian distribution with mean 0 and variance vary.
    :param vary:
    :return:
    """
    # first, set up the weight init function
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0.0, vary)

    return weights_init_normal
