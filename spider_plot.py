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

"""General Settings"""
pickle_name_models = f'models_all_experiments_volta'
pickle_name_sims = f'pairwise_all_experiment_volta'
n = 4


"""Create all the settings for a bunch of models"""
# number of classes
class_lists = [list(range(k)) for k in range(5, 10)][:n]
n_settings = [pert.PerturbationSettings(unmod_classes=k) for k in class_lists]

# column masking
column_lists = [list(range(k)) for k in range(0, 34, 4)]
col_settings = [pert.PerturbationSettings(unmod_classes=[], 
                                          mod_classes=list(range(5)), 
                                          columns=k) for k in column_lists]

# gaussian noise
std_list = [0] + [2**k for k in (-4, -3, -2, -1, 0, 1, 2)][:n]
var_list = [k**2 for k in std_list]
noise_settings = [pert.PerturbationSettings(unmod_classes=[], 
                                            mod_classes=list(range(5)),
                                            noise=True, var=k) 
                                            for k in var_list]

# image mixing
image_count = np.geomspace(50, 5000, 8)
image_count = [0] + [int(np.ceil(k)) for k in image_count][:n-1]
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
swap_loader_settings = zip(swap_settings, swap_list)

all_settings = n_settings + col_settings + noise_settings + swap_loader_settings

"""Train each of the models"""
model_list = []
for setting in all_settings:
    if hasattr(setting, '__len__'):
        loaders = pert.subset_class_loader(setting[0], swap=setting[1])
    else:
        loaders = pert.subset_class_loader(setting)
    
    # create the model
    model = spec.spectrum_analysis([512])
    model.train(loaders[0], loaders[1], 5, grain=10000, ep_grain=5)
    model_list.append(model)
    del model

# save the model list
# with open(f'pickle_archive/{pickle_name_models}.pkl', 'wb') as file:
with open(f'{pickle_name_models}.pkl', 'wb') as file:
    pickle.dump(model_list, file)
os.system('say "saved trained models to disc"')


"""Calculate the Pairwise Similarities"""
similarities = pm.compute_pairwise_sims(model_list)
# with open(f'pickle_vars/pairwise_sims/{pickle_name_sims}.pkl', 'wb') as file:
with open(f'{pickle_name_sims}.pkl', 'wb') as file:
    pickle.dump(similarities, file)
os.system('say "finished calculating pairwise similarities and saved to disc"')
