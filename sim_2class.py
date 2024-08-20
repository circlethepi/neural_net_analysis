"""Imports"""
import neural_network as nn_mod
import spectral_analysis as spec
import network_similarity as sim
# import class_splitter as cs
# import distance_mapping as dm
import perturbation as pert
import perturbation_to_map as pm
import utils

# basic needs packages
import numpy as np
import torch
from torchvision import datasets
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits import mplot3d
import torchvision
import pickle
from fractions import Fraction
from tqdm import tqdm

import argparse
import os

"""Arguments for Command Line"""
parser = argparse.ArgumentParser()

### Adding Arguments

## Naming Settings
parser.add_argument('--save_suffix', type=str, 
                    help='suffix to generated name for save file')

## 2 class w reference similarities
parser.add_argument('--is2class', action='store_true', 
                    help='Whether this is is a 2class similarity experiment')
parser.add_argument('--class_ind', type=int, help="index of reference class")
parser.add_argument('--arch', type=str, help='architecture of the expiriment')
parser.add_argument('--dataset', type=str, help='dataset for experiment')
parser.add_argument('--epstep', type=int, help='step for epochs in folder')

## General Similarities Settings
parser.add_argument('--w_clip', default=30, type=int, help="weight clip rank \
                    for metric calculation")
parser.add_argument('--a_clip', default=64, type=int, help="activcation clip \
                    rank for metric calculation")
parser.add_argument('--distance', action='store_false', help="sets sim to BW \
                    distance instead")
parser.add_argument('--model_inds1', nargs='+', type=int, default=None,
                    help='must take in 2 values, start and end')
parser.add_argument('--model_inds2', nargs='+', type=int, default=None,
                    help='must take in 2 values, start and end')
parser.add_argument('--sim_sec', type=str, default='full',
                    help='naming for section of sim matrix if broken')

# DIRECTORY NAMES
parser.add_argument('--mod_dir', default='model_library', help="name of dir to\
                     store/load trained models")
parser.add_argument('--acc_dir', default='acc_library', help="name of dir to \
                    store/load trained model accuracies")
parser.add_argument('--sim_dir', default='sim_library', help="name of dir to \
                    store/load pairwise sims")
parser.add_argument('--rel_path', type=str, default='.', 
                    help='relative path to mod/acc/sim directories')


args = parser.parse_args()
# get directory settings
mod_dir = args.mod_dir
acc_dir = args.acc_dir
sim_dir = args.sim_dir

rel_path = args.rel_path

# general similarity settings
w_clip = args.w_clip
a_clip = args.a_clip
do_sim = args.distance
inds_models1 = args.model_inds1
inds_models2 = args.model_inds2
class_ind = args.class_ind
dataset = args.dataset
arch = args.arch
epstep = args.epstep
sim_sec = args.sim_sec
save_suff = args.save_suffix if args.save_suffix is not None else ""


if args.is2class:
    assert class_ind is not None, 'Cannot do 2 class with no specified ref class'


"""Set-Up"""
class_others = list(range(10))
class_others.pop(class_ind)
class_others = class_others

cifar_classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 
            'Ship', 'Truck')
dataset_dict = {'mnist' : datasets.MNIST, 'cifar' : datasets.CIFAR10}
class_dict = {'mnist' : list(range(10)), 'cifar' : cifar_classes}
input_size_dict = {'mnist' : 28*28, 'cifar' : 32*32*3}

dataset_class = dataset_dict[dataset]
classes = class_dict[dataset]
input_size = input_size_dict[dataset]

experiment_name = f'{dataset}_2class_{class_ind}_all'

# function to deal with different batch numbers for each of the models
def extract_epoch_name(filename):
    name_part = os.path.splitext(filename)[0]  # Remove the file extension
    if '-' in name_part:
        numerator, denominator = map(int, name_part.split('-'))
        return numerator / denominator
    else:
        return int(name_part)


"""Getting the Loader"""
class_list = class_others + [class_ind]
setting = pert.PerturbationSettings(unmod_classes=class_list, 
                                    dataset_class=dataset_class)
train_loader, _ = pert.subset_class_loader(setting)


"""Import Each of the Models"""
model_list = []
for k in range(len(class_others)):
    # make the epoch list 
    dirpath = f'{rel_path}/model_library/{experiment_name}'
    filepath = f'{dirpath}/{experiment_name}_{class_others[k]}-{arch}'

    all_files = os.listdir(filepath)
    epochs = [os.path.splitext(file)[0] for file in all_files]

    epoch_list = sorted(epochs, key=extract_epoch_name)

    print('Total Number of Epochs Recorded: ', len(epoch_list))
    print('Number of Epochs to Use: ', len(epoch_list[::epstep]))
    
    for ep in epoch_list[::epstep]:
        # load the model
        model = spec.spectrum_analysis([512], load=True, path=filepath, 
                                       epoch=ep, seed=None, input_size=input_size)
        # set the loader
        #model.set_train_loader(train_loader)

        # add to the list of models
        model_list.append(model)
        del model
print(f'Total Number of Models: {len(model_list)}')

"""Calculate the Similarities"""
#print(inds_models1, inds_models2)
model_list1 = model_list[inds_models1[0]:inds_models1[1]]
model_list2 = model_list[inds_models2[0]:inds_models2[1]]
# if model_list1 == model_list2:
#     model_list2 = None
mnist_sims = pm.compute_pairwise_sims(model_list1, train_loader,
                                      model_set2=model_list2)

"""Save the Calculated Similarities"""
filename = f'{rel_path}/sim_library/{experiment_name}_({sim_sec}){save_suff}_similarities.pkl'
with open(filename, 'wb') as file:
    pickle.dump(mnist_sims, file)

print("SAVED AND COMPLETE") 

