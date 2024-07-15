import neural_network as nn_mod
import spectral_analysis as spec
import network_similarity as sim
import class_splitter as cs
import distance_mapping as dm
import perturbation as pert
import perturbation_to_map as pm

import numpy as np
import pickle
import os

import argparse

"""
To-Do:
- add a way to take certain indices from a loaded model
- (can save this subset to model library to use, or directly begin using)
"""


"""Arguments for Command Line"""
parser = argparse.ArgumentParser()

# adding arguments
parser.add_argument('--name', type=str, required=True, 
                    help="name of the experiment/file to save to")
# experiments it could run
parser.add_argument('--nclass', action='save_true', help="whether do n classes")
parser.add_argument('--cols', action='save_true', help="whether do col masking")
parser.add_argument('--gnoise', action='save_true', help="whether do gnoise")
parser.add_argument('--mixing', action='save_true', help="whether mix images")

# names of trained models to load if that's a thing
parser.add_argument('--load_model_names', nargs='+')

# whether to calculate pairwise similarities
parser.add_argument('--do_sims', action='save_true', help="whether calculate \
                    pairwise sims")

# directory names
parser.add_argument('--mod_dir', default='model_library', help="name of dir to\
                     store/load trained models")
parser.add_argument('--acc_dir', default='acc_library', help="name of dir to \
                    store/load trained model accuracies")
parser.add_argument('--sim_dir', default='sim_library', help="name of dir to \
                    store/load pairwise sims")

args = parser.parse_args()


"""General Settings"""
# setting the name of the files
pickle_name_models = args.name
pickle_name_accuracies = f'{args.name}_accuracies'
pickle_name_sims = f'{args.name}_pairwise'

mod_dir = args.mod_dir
acc_dir = args.acc_dir
sim_dir = args.sim_dir

# parsing the model settings



"""Create all the settings for a bunch of models"""
# number of classes
class_lists = [list(range(k)) for k in range(5, 10)]
n_settings = [pert.PerturbationSettings(unmod_classes=k) for k in class_lists]

# column masking
column_lists = [list(range(k)) for k in range(0, 32)]
col_settings = [pert.PerturbationSettings(unmod_classes=[], 
                                          mod_classes=list(range(5)), 
                                          columns=k) for k in column_lists]

# gaussian noise
std_list = [0] + [2**k for k in range(-10, 3)]
var_list = [k**2 for k in std_list]
noise_settings = [pert.PerturbationSettings(unmod_classes=[], 
                                            mod_classes=list(range(5)),
                                            noise=True, var=k) 
                                            for k in var_list]

# image mixing
image_count = np.geomspace(50, 5000, 10)
image_count = [0] + [int(np.ceil(k)) for k in image_count]
class_list = list(range(10))
swap_list = [[pert.SwapSettings(5, 0, ec, bidirectional=True, old_outside=True),
              pert.SwapSettings(6, 1, ec, bidirectional=True, old_outside=True),
              pert.SwapSettings(7, 2, ec, bidirectional=True, old_outside=True),
              pert.SwapSettings(8, 3, ec, bidirectional=True, old_outside=True),
              pert.SwapSettings(9, 4, ec, bidirectional=True, old_outside=True)] 
              for ec in image_count] 
swap_settings = [pert.PerturbationSettings(unmod_classes=[], 
                                           mod_classes=class_list) for 
                 k in range(len(image_count))]
swap_loader_settings = list(zip(swap_settings, swap_list))

setting_list = [n_settings, col_settings, noise_settings, swap_loader_settings]

all_settings = []

# setting the settings because yay
for i in range(4):
    k = [args.nclass, args.cols, args.gnoise, args.mixing][i]
    if k:
        all_settings.append(setting_list[i])


"""Train each of the models as specified"""
if all_settings:
    model_list = []
    accuracies = []
    for setting in all_settings:
        cs.set_seed(1234)
        if hasattr(setting, '__len__'):
            loaders = pert.subset_class_loader(setting[0], swap=setting[1])
        else:
            loaders = pert.subset_class_loader(setting)
        
        # create the model
        cs.set_seed(1234)
        model = spec.spectrum_analysis([512])
        cs.set_seed(1234)
        model.train(loaders[0], loaders[1], 5, grain=10000, ep_grain=5)
        model_list.append(model)
        accuracies.append(model.val_history[-1])
        del model

    # save the model list
    with open(f'{mod_dir}/{pickle_name_models}.pkl', 'wb') as file:
    #with open(f'{pickle_name_models}.pkl', 'wb') as file:
        pickle.dump(model_list, file)
    announcement_string = f'saved {args.name} models to disc'
    os.system(f'say {announcement_string}')

    # save the accuracies list
    with open(f'{acc_dir}/{pickle_name_accuracies}.pkl', 'wb') as file:
        pickle.dump(accuracies, file)
    announcement_string = f'saved {args.name} accuracies to disc'
    os.system(f'say {announcement_string}')


if args.load_model_names:
    # load from the model dir given
    # load each model name
    # add each to a list and call it model list
    model_list = []
    for model_name in args.load_model_names:
        with open(f'{mod_dir}/{model_name}.pkl', 'rb') as file:
            model_list_constituent = pickle.load(file)
        model_list += model_list_constituent
    
"""Calculate the Pairwise Similarities"""
if args.do_sims:
    similarities = pm.compute_pairwise_sims(model_list)
    with open(f'{sim_dir}/{pickle_name_sims}.pkl', 'wb') as file:
    #with open(f'{pickle_name_sims}.pkl', 'wb') as file:
        pickle.dump(similarities, file)
    os.system('say "finished calculating pairwise similarities and saved to disc"')
