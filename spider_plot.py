import neural_network as nn_mod
import spectral_analysis as spec
import network_similarity as sim
import class_splitter as cs
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

## adding arguments
parser.add_argument('--name', type=str, required=True, 
                    help="name of the experiment/file to save to")
# RUNNING EXPERIMENTS (preset, small)
parser.add_argument('--nclass', action='store_true', help="whether do n classes")
parser.add_argument('--cols', action='store_true', help="whether do col masking")
parser.add_argument('--gnoise', action='store_true', help="whether do gnoise")
parser.add_argument('--mixing', action='store_true', help="whether mix images")

# LOADING TRAINED MODELS
parser.add_argument('--load_model_names', nargs='+', default=None)
# trained model index selections for up to 4 models 
parser.add_argument('--exp_indices_1', nargs='+', type=int, default=None)
parser.add_argument('--exp_indices_2', nargs='+', type=int, default=None)
parser.add_argument('--exp_indices_3', nargs='+', type=int, default=None)
parser.add_argument('--exp_indices_4', nargs='+', type=int, default=None)

# SIMILARITIES
parser.add_argument('--do_sims', action='store_true', help="whether calculate \
                    pairwise sims")
parser.add_argument('--w_clip', default=30, type=int, help="weight clip rank \
                    for metric calculation")
parser.add_argument('--a_clip', default=64, type=int, help="activcation clip \
                    rank for metric calculation")
parser.add_argument('--distance', action='store_false', help="sets sim to BW \
                    distance instead")
parser.add_argument('--side_sims_2', action='store_true', help="whether to \
                    treat loaded model lists as two sides of the matrix of 1")

# DIRECTORY NAMES
parser.add_argument('--mod_dir', default='model_library', help="name of dir to\
                     store/load trained models")
parser.add_argument('--acc_dir', default='acc_library', help="name of dir to \
                    store/load trained model accuracies")
parser.add_argument('--sim_dir', default='sim_library', help="name of dir to \
                    store/load pairwise sims")

## pargs the args
args = parser.parse_args()



"""General Settings from Parser"""
# setting the name of the files
pickle_name_models = args.name
pickle_name_accuracies = f'{args.name}_accuracies'
pickle_name_sims = f'{args.name}_pairwise'

mod_dir = args.mod_dir
acc_dir = args.acc_dir
sim_dir = args.sim_dir

# parsing the model settings

# parsing the similarity settings
w_clip = args.w_clip
a_clip = args.a_clip
do_sim = args.distance




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
        all_settings += setting_list[i]


"""Train each of the models as specified"""
if all_settings:
    print('Number of Models to Train: ', len(all_settings))
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
    print(announcement_string)
    #os.system(f'say {announcement_string}')

    # save the accuracies list
    with open(f'{acc_dir}/{pickle_name_accuracies}.pkl', 'wb') as file:
        pickle.dump(accuracies, file)
    announcement_string = f'saved {args.name} accuracies to disc'
    print(announcement_string)
    #os.system(f'say {announcement_string}')


""" Loading in Models as Specified """
# first check to see if there are indices given 
model_idex_lists = [args.exp_indices_1, args.exp_indices_2, args.exp_indices_3,
                    args.exp_indices_4]
not_null = [i for i,v in enumerate(model_idex_lists) if v != None]

# check to see if all the indices that have lists are valid
if not_null:
    assert max(not_null) <= (len(args.load_model_names)-1) if \
        hasattr(args.load_model_names, '__len__') else max(not_null) == 1, \
        'max index k for exp_indices_k must be < # model lists imported'

if args.load_model_names is not None:
    # load from the model dir given
    # load each model name
    # add each to a list and call it model list
    if not args.side_sims_2:
        model_list = []
        i = 0
        for model_name in args.load_model_names:
            with open(f'{mod_dir}/{model_name}.pkl', 'rb') as file:
                model_list_constituent = pickle.load(file)
            # if importing a single model
            if not hasattr(model_list_constituent, '__len__'):
                model_list_constituent = [model_list_constituent]
            
            # if this set of models has index selection
            if i in not_null:
                this_model_inds = model_idex_lists[i]
                # check to make sure the max index is valid
                assert max(this_model_inds) <= len(model_list_constituent) -1,\
                f'invalid index selection for {i+1}th model list'

                # if it is valid, then do the index selection
                model_list_constituent = [model_list_constituent[k] for k in \
                                          this_model_inds]

            model_list += model_list_constituent
            i += 1
    else: 
        if hasattr(args.load_model_names, '__len__'):
            if len(args.load_model_names) != 2:
                raise Exception(f'For 2-sided similarity calculation, the \
                                number of models loaded must\nbe exactly 2')
        model_sets = []
        for model_name in args.load_model_names:
            with open(f'{mod_dir}/{model_name}.pkl', 'rb') as file:
                model_list_constituent = pickle.load(file)
            model_sets.append(model_list_constituent)
    

"""Calculate the Pairwise Similarities"""
if args.do_sims:
    if not args.side_sims_2:
        similarities = pm.compute_pairwise_sims(model_list, similarity=do_sim,
                                                w_clip=w_clip, a_clip=a_clip)
    else:
        similarities = pm.compute_pairwise_sims(model_sets[0], similarity=do_sim,
                                                w_clip=w_clip, a_clip=a_clip,
                                                model_set2=model_sets[1])
    
    with open(f'{sim_dir}/{pickle_name_sims}.pkl', 'wb') as file:
    #with open(f'{pickle_name_sims}.pkl', 'wb') as file:
        pickle.dump(similarities, file)
    announcement_string = f'finished calculating pairwise similarities and saved'
    print(announcement_string)
    #os.system('say "finished calculating pairwise similarities and saved to disc"')
