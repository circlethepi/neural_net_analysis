import sys
sys.path.insert(1, '../')
# set CUDA device
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"

#import sys
#sys.path.insert(1, '../')

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
import matplotlib as mpl
from mpl_toolkits import mplot3d
import torchvision
import pickle
from fractions import Fraction
from tqdm import tqdm
import json

# get the dataloaders
train, test = nn_mod.get_dataloaders()

# set checkpoints
# want every 5 epochs, first 5, and grain in 1
image_counts = [64*k for k in (0, 5, 14, 38, 104, 285)] + \
                [50000*k for k in range(1,51)] 

n_eps, checkpoints = spec.calculate_epochs_checkpoints(image_counts, 64, 50000)

arches_train = [[2, 64], [1, 64], [256, 64], [64]]
# 1-64 is basically a single-layer network but with a relu 

# make the model for fc64fc64
# make the model
for arch in arches_train:
    model = spec.SpectrumAnalysis(arch, seed=0, exp_name='cifar_init0_dict', save=True, rel_path='')
    # train the model
    model.train(train, test, n_eps, checkpoints=checkpoints, save=True)


### GET THE SAVED PERFORMANCE
filename = f'pickle_vars/cifar_2layer_size_perf_all.json'

with open(filename, 'r') as file:
    to_save = json.load(file)

perf_dict = to_save['test']
perf_dict2 = to_save['train']

# GET THE SAVED CHECKPOINTS
model_loc = f'{spec.model_savedir}/cifar_init0-fc32fc64'

# get sorted model checkpoints
epoch_list = utils.get_sorted_epoch_names(model_loc)
checkpoints = [utils.parse_frac_string(ep) for ep in epoch_list]


# add the new [64,64] model to the dictionaries
for arch in arches_train:
    # arch = [64,64]
    perf_test = []
    perf_train = []
    arch_strings = [f'fc{k}' for k in arch]
    suff = ''.join(arch_strings)  

    model_loc = f'{spec.model_savedir}/cifar_init0_dict-{suff}'

    if len(arch) == 1:
        keykey = '1layer'
    else:
        keykey = arch[0]

    for ep in epoch_list:
        model = spec.SpectrumAnalysis(arch, load=True, path=model_loc, epoch=ep)
        perf_test.append(model.evaluate_model(test))
        perf_train.append(model.evaluate_model(train))
    perf_dict[keykey] = perf_test.copy()
    perf_dict2[keykey] = perf_train.copy()


to_save_new = dict(zip(('test', 'train', 'checkpoints'), (perf_dict, perf_dict2, checkpoints)))
filename = 'pickle_vars/cifar_2layer_size_perf_all_1thru256.json'
with open(filename, 'w') as file:
    json.dump(to_save_new, file)
