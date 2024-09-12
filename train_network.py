# training the network
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
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets

# basics
import numpy as np
from matplotlib import pyplot as plt
from fractions import Fraction as frac

# other packages/files
import neural_network
from utils import *

# check if there is a GPU available
device = set_torch_device()

###############################
# Training the Neural Network #
###############################


def update_spectrum(model_name, spectrum_list, var_hist, cov_list):
    """

    :param model_name:      the name of the neural network model
    :param spectrum_list:   the existing list of spectra
    :param var_hist:        the variance history list
    :param cov_list:        the covariance history list
    :return:                updated versions of each of the lists
    """
    n_neurons = model_name.n_neurons
    # print("n_layers: ", model_name.n_layers)
    weight_list = []
    for layer in model_name.layers:
        weight_list.append(layer.weight.numpy(force=True))

    spec_to_add = []
    var_to_add = []
    cov_to_add = []

    for i in range(len(weight_list)-1):
        # print(i)  # sanity check
        weights = weight_list[i]
        # getting the covariance
        c = weights @ weights.transpose() * (1/n_neurons[i])
        eigenvals, eigenvecs = np.linalg.eigh(c)
        spec_to_add.append(eigenvals[::-1])
        var_to_add.append(np.sum(eigenvals))
        cov_to_add.append(c)

    # adding to the list
    spectrum_list.append(spec_to_add)
    var_hist.append(var_to_add)
    cov_list.append(cov_to_add)

    return spectrum_list, var_hist, cov_list


def val_accuracy(model_name, criterion, val_loader, val_loss_history, val_acc_history):
    # print('testing accuracy')
    # set the model to evaluation mode
    model_name.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_name(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.argmax(1) == labels).sum().item()

    # calculate the average validation loss and accuracy
    val_loss /= len(val_loader)
    val_loss_history.append(val_loss)
    val_acc /= len(val_loader.dataset)
    val_acc_history.append(val_acc)

    return val_loss_history, val_acc_history


def evaluate_model(model_name, val_loader):
    model_name.eval()
    val_acc = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model_name(inputs)
            val_acc += (outputs.argmax(1) == labels).sum().item()
    val_acc /= len(val_loader.dataset)

    return val_acc


def train_accuracy(model_name, optimizer, criterion, train_loader, train_loss_history, train_acc_history):
    train_loss = 0
    train_acc = 0
    # set model to train mode
    model_name.train()
    # iterate over the training data
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model_name(inputs)
        # compute the loss
        loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()
        # increment the running loss and accuracy
        train_loss += loss.item()
        train_acc += (outputs.argmax(1) == labels).sum().item()

    # calculate the average training loss and accuracy
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    train_acc /= len(train_loader.dataset)
    train_acc_history.append(train_acc)

    return train_loss_history, train_acc_history


def train_model(model_name, train_loader, val_loader, n_epochs,
                grain=10, ep_grain:int=2,
                criterion=nn.CrossEntropyLoss(), save=False, savepath=None,
                checkpoints=None):

    """
    trains a single-layer model for a single epoch and records the val and train accuracy
    Also records the spectrum as we go
    Inputs
    model_name : single_layer     neural network model to train
    grain      : int              how often to check accuracy in log scale
    checkpoints: dict[epoch : int] -> list(batches : int)  when to check accuracy/
                                  saved model state at specific intervals
    """
    #################
    #### Set-up #####
    #################
    if save:
        assert savepath is not None, 'To save, there must be a save path'
        save_model(model_name, 0, savepath)

    ep_history = []

    # setting the optimizer
    optimizer = torch.optim.Adam(model_name.parameters(), lr=0.001)

    total_batches = len(train_loader)

    # set when to evaluate the model state

    if checkpoints is None:
        # if no specific checkpoints, then use the grain settings
        # setting the values for which we check the performance
        max_it = int(np.floor(np.emath.logn(grain, total_batches)))  # the max n such that grain^n < total
        # creating the list of iterations at which to check the performance
        intervals = [grain**i for i in range(1, max_it + 1)]
        # intervals.append(total)
        if total_batches not in intervals:
            intervals.append(total_batches)
        print(f'testing at {intervals} batches of {total_batches} total batches')

        # set the epochs at which to check the performance
        assert ep_grain >= 1, 'Epoch grain must be >= 1'
        if ep_grain < n_epochs and ep_grain != 1:
            max_ep = int(np.floor(np.emath.logn(ep_grain, n_epochs)))
            ep_intervals = [ep_grain**i for i in range(1, max_ep+1)]
        elif ep_grain == 1:
            ep_intervals = list(range(2, n_epochs))
        else:
            ep_intervals = []
        if n_epochs not in ep_intervals and n_epochs > 1:
            ep_intervals.append(n_epochs)
    
    if checkpoints is not None:
        # if there are specific checkpoints, then set them
        if 0 in checkpoints.keys():
            intervals = checkpoints[0]
        else:
            intervals = []

        # check to see if there are specific epochs
        ep_intervals = []
        for ep in checkpoints.keys():
            if 0 in checkpoints[ep]:
                ep_intervals.append(ep)
        if n_epochs not in ep_intervals and n_epochs > 1:
            ep_intervals.append(n_epochs)
    
    print(f'testing at {ep_intervals} epochs of {n_epochs} total epochs')

    # set tracking
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    # also tracking the spectrum as we go
    spectrum_history = []
    total_var_history = []
    # and the covariance
    cov_history = []

    # UPDATING the things
    # updating the epoch list
    #print(f'getting spectrum at init...')
    ep_history.append(0)
    # updating the spectrum
    spectrum_history, total_var_history, cov_history = update_spectrum(model_name, spectrum_history, total_var_history,
                                                                       cov_history)
    # updating the test history
    val_loss_history, val_acc_history = val_accuracy(model_name, criterion, val_loader, val_loss_history,
                                                     val_acc_history)
    # updating the train history
    train_loss_history, train_acc_history = train_accuracy(model_name, optimizer, criterion, train_loader,
                                                           train_loss_history, train_acc_history)

    ########################################
    ##### Training the Model (EPOCH 1) #####
    ########################################

    # set model to train mode
    model_name.train()

    i = 1
    # iterate over the training data
    for inputs, labels in tqdm(train_loader, desc='Training epoch 1'):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model_name(inputs)
        # compute the loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i in intervals:
            #print(f'testing accuracy, calculating spectrum after {i}th batch')
            ep_history.append('{:.2e}'.format(float(frac(i, total_batches))))
            # updating the spectrum
            spectrum_history, total_var_history, cov_history = update_spectrum(model_name, spectrum_history,
                                                                               total_var_history, cov_history)
            # updating the test history
            val_loss_history, val_acc_history = val_accuracy(model_name, criterion, val_loader, val_loss_history,
                                                             val_acc_history)
            # updating the train history
            train_loss_history, train_acc_history = train_accuracy(model_name, optimizer, criterion, train_loader,
                                                                   train_loss_history, train_acc_history)
            if save:
                epoch = f'{i}-{total_batches}'
                if i == total_batches:
                    epoch = int(1)
                save_model(model_name, epoch, savepath)
                
        # increasing the step
        i += 1

    #print(f'Training of first epoch complete.\nTraining for the next {n_epochs-1}')

    ###############################################
    ### training the model for the other epochs ###
    ###############################################
    # train over the epochs
    if n_epochs > 1:
        for epoch in range(2, n_epochs+1):
            train_loss = 0.0
            train_acc = 0.0

            # get the batches for the intermediate steps
            if epoch-1 in checkpoints.keys():
                intermediate_intervals = checkpoints[epoch-1]
            else:
                intermediate_intervals = []

            # set model to train mode
            model_name.train()

            i = 1
            # iterate over the training data
            for inputs, labels in tqdm(train_loader, desc=f'Training epoch {epoch}'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model_name(inputs)
                # compute the loss
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # increment the running loss and accuracy
                train_loss += loss.item()
                train_acc += (outputs.argmax(1) == labels).sum().item()

                if i in intermediate_intervals:
                    #print(f'testing accuracy, calculating spectrum after {i}th batch')
                    ep_history.append('{:.2e}'.format(float(frac(((total_batches*(epoch-1)))+i, total_batches))))
                    # updating the spectrum
                    spectrum_history, total_var_history, cov_history = update_spectrum(model_name, spectrum_history,
                                                                                    total_var_history, cov_history)
                    # updating the test history
                    val_loss_history, val_acc_history = val_accuracy(model_name, criterion, val_loader, val_loss_history,
                                                                    val_acc_history)
                    # updating the train history
                    train_loss_history, train_acc_history = train_accuracy(model_name, optimizer, criterion, train_loader,
                                                                        train_loss_history, train_acc_history)
                    if save:
                        epoch_state = f'{(total_batches*(epoch-1))+i}-{total_batches}'
                        save_model(model_name, epoch_state, savepath)
                i += 1

            if epoch in ep_intervals:
                #print(f'epoch {epoch} getting spectrum')
                ep_history.append(epoch)
                # UPDATING the things
                spectrum_history, total_var_history, cov_history = update_spectrum(model_name, spectrum_history,
                                                                                total_var_history, cov_history)
                # updating the test history
                val_loss_history, val_acc_history = val_accuracy(model_name, criterion, val_loader, val_loss_history,
                                                                val_acc_history)
                # updating the train history
                train_loss_history, train_acc_history = train_accuracy(model_name, optimizer, criterion, train_loader,
                                                                    train_loss_history, train_acc_history)
                if save:
                    save_model(model_name, epoch, savepath)

        # getting the last epoch
        if epoch not in ep_intervals:
            #print(f'epoch {epoch} getting spectrum (last)')
            ep_history.append(epoch)
            # UPDATING the things
            spectrum_history, total_var_history, cov_history = update_spectrum(model_name, spectrum_history,
                                                                            total_var_history, cov_history)
            # updating the test history
            val_loss_history, val_acc_history = val_accuracy(model_name, criterion, val_loader, val_loss_history,
                                                            val_acc_history)
            # updating the train history
            train_loss_history, train_acc_history = train_accuracy(model_name, optimizer, criterion, train_loader,
                                                                train_loss_history, train_acc_history)
            if save:
                save_model(model_name, epoch, savepath)

    ##########################
    ### setting up returns ###
    ##########################
    return ep_history, val_acc_history, train_acc_history, spectrum_history


def save_model(model, epoch, path):
    filename = f'{path}/{epoch}.pt'

    torch.save(model, filename)

    print(f'Model saved to disc, epoch {epoch}')

    return