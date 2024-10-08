import spectral_analysis as spec
import alignment as align
import network_similarity as sim
#import archive.class_splitter as cs
import neural_network as nn_mod
from utils import *

import pickle
from tqdm import tqdm
#from tqdm.notebook import tqdm
import time
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy
import torch
import torchvision
import torch.utils.data.dataset
from torchvision import datasets
from torchvision import transforms

########################
### GLOBAL VARIABLES ###
########################
# baseline measurements from pickles
# load in the baseline identical measurements
#
# with open('/Users/mnzk/Documents/40-49. Research/42. Nets/42.97. Library/pickle_vars/baseline_identical.pkl', 'rb') as f:
# with open('../pickle_vars/baseline_identical.pkl', 'rb') as f:
#     id_baseline = pickle.load(f)
# way_id_sim = id_baseline['way_sim']
# act_id_sim = id_baseline['act_sim']
# way_id_dist = id_baseline['way_dist']
# act_id_dist = id_baseline['act_dist']

# # load in the quantity baselines
# # with open('/Users/mnzk/Documents/40-49. Research/42. Nets/42.97. Library/pickle_vars/baseline_quantiles.pkl', 'rb') as f:
# with open('../pickle_vars/baseline_quantiles.pkl', 'rb') as f:
#     quantities = pickle.load(f)

# class names
class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 
               'ship', 'truck')

# Figure font sizes
axis_fontsize=18
title_fontsize=18

##### Setting the random seed
COMMON_SEED = 1234

########################################
### Perturbation Object and Settings ###
########################################
class SwapSettings:
    def __init__(self, old_label, new_label, n_images, bidirectional=False, 
                 old_outside=False):
        """

        :param old_label: int   :   the old class label that will be changed on 
                                    the images (the images getting "mixed in"
                                    to the new label)
        :param new_label: int   :   the new class label for the images getting 
                                    "mixed in"
        :param n_images: int    :   the number of images to change the label 
                                    for (the number of images to "mix in")
        :param bidirectional: bool  :   whether to make changes to both classes 
                                        (trading images between them)
        :param old_outside:   bool  :   whether the old label images are 
                                        outside of the dataset otherwise 
                                        (whether the old label is one of the 
                                        classes used in training)
        """
        self.old_label = old_label
        self.new_label = new_label
        self.n_images = n_images
        self.bidirectional = bidirectional
        self.old_outside = old_outside
        return


class PerturbationSettings:
    """
    Holds the information for a call of subset_class_loader or for an 
    initialization of a Perturbation experiment.
    """
    def __init__(self, columns=None, rows=None, val=(1,1,1), intensity=False, 
                 noise=False, var=None, random_pixels=None,
                 unmod_classes=tuple(range(10)), mod_classes=None, name=None,
                 batch_size=64, dataset_class=datasets.CIFAR10, 
                 class_count=None, random_subsets:bool=False, resize=None,
                 greyscale=False):
        """

        :param columns: list() or list(list))   :   the columns or list of settings 
                                                    for columns to be masked
        :param rows:                            :   same as for columns
        :param val: tuple(float, float, float)  :   RGB value for masking
        :param intensity: bool                  :   whether the experiment 
                                                    should cut off values of 
                                                    intensity
        :param noise: bool                      :   whether to add noise to the 
                                                    perturbation
        :param var: float or list(float)        :   the value(s) of the variance 
                                                    for when noise is true
        :param random_pixels: DO NOT USE THIS< IT DOES NOT WORK
        :param unmod_classes: list(int) or list(list(int))  : list of class indices 
                                                            to load un-modified
        :param mod_classes:             same as above, but for modified classes
        :param name: str                        :   the name of the experiment
        :param batch_size: int                  :   the size of batch to use in 
                                                    the dataloader
        :param class_count: int or list(int)    :   sizes of each class in order
        :param random_subsets: bool             :   used with class_count. When
                                                    set to True, images will be
                                                    chosen randomly. When False,
                                                    images are picked in order.
                                                    If class_count is None and
                                                    random subsets is True, this
                                                    has no effect
        """
        self.columns = columns 
        self.rows = rows
        self.val = val

        self.intensity = intensity

        self.noise = noise
        self.var = var

        self.random_pixels = random_pixels

        self.unmod_classes = unmod_classes
        self.mod_classes = mod_classes
        self.name = name

        self.batch_size = batch_size
        self.dataset_class = dataset_class


        if class_count and mod_classes and\
        len(class_count) != len(mod_classes) \
        and len(class_count) != 1:
            raise Exception(f'Number of images per class is different from the number of classes included.')
        elif class_count and mod_classes and len(class_count) == 1:
            class_count = class_count * len(mod_classes)
            #print(f'len class count list', len(class_count))
        
        self.class_count = class_count

        self.random_subsets = random_subsets

        self.resize = resize
        self.greyscale = greyscale

        return


default_perturbation_settings = PerturbationSettings()


# class PerturbationResults:
#     """
#     holds and plots perturbation experiment results. These should be reshaped to be indexed by layer!
#     """
#     # def __init__(self, *initial_data, **kwargs):
#     #     for dictionary in initial_data:
#     #         for key in dictionary:
#     #             setattr(self, key, dictionary[key])
#     #     for key in kwargs:
#     #         setattr(self, key, kwargs[key])

#     def __init__(self, results_dict):
#         """

#         :param results_dict:
#         """
#         self.similarities = results_dict['sims_u']
#         self.similarities_clipped = results_dict['sims_c']

#         self.distances = results_dict['dist_u']
#         self.distances_clipped = results_dict['dist_c']

#         self.accuracy = results_dict['acc']
#         self.accuracy_baseline = results_dict['acc_base']
#         self.dimensions_trials = results_dict['effdims']

#         self.dimensions_baseline = results_dict['base_dims']

#         self.baseline_trace = results_dict['base_tr_u']
#         self.baseline_trace_clipped = results_dict['base_tr_c']

#         self.experiment_trace = results_dict['exp_tr_u']
#         self.experiment_trace_clipped = results_dict['exp_tr_c']

#         self.nuclear_norm = results_dict['nnorm_u']
#         self.nuclear_norm_clipped = results_dict['nnorm_c']

#         self.description = results_dict['README']

#         self.models = results_dict['models']

#         self.ticks = None

#     def set_ticks(self, ticklist):
#         setattr(self, 'ticks', ticklist)
#         print('Successfully set the ticks for the model')
#         return

#     def plot_trajectories(self, layer=1, ticks=None, ylog=True, xlog=False,
#                           xlabadd='', plot_baselines=True, ylims=None):
#         ticks = self.ticks if self.ticks else ticks
#         if not ticks:
#             os.system('say "PLEASE SET THE TICKS YOU ABSOLUTE POTATO"')
#             raise Exception('Need to set ticks before plotting')

#         plot_result_trajectories(self.similarities[layer], 
#                                  self.similarities_clipped[layer],
#                                  self.distances[layer], 
#                                  self.distances_clipped[layer],
#                                  xticks=ticks, ylog=ylog, xlog=xlog, 
#                                  xlabadd=xlabadd, plot_baselines=plot_baselines,
#                                  title=f'Layer {layer}', ylims=ylims)
#         return

#     def plot_trace_nnorms(self, ticks=None, quantity='weights', layer=1, clipped=True,
#                           titleadd='', xlabel='Perturbation Level', upper_legend_loc='best', lower_legend_loc='best',
#                           yrange_weights=None, yrange_activations=None,
#                           xlog=False, hline_lims=None, ylog=True):
#         ticks = self.ticks if self.ticks else ticks
#         if ticks is None:
#             os.system('say "PLEASE SET THE TICKS YOU ABSOLUTE POTATO"')
#             raise Exception('Need to set ticks before plotting')

#         plot_trace_nnorms(self, ticks, layer=layer, clipped=clipped, xlabel=xlabel, yrange_weights=yrange_weights, yrange_activations=yrange_activations,
#                           xlog=xlog, ylog=ylog, hline_lims=hline_lims)

#         return

#     def plot_accuracy(self, type='test', ticks=None, titleadd='', 
#                       legend_loc='lower left', xlabel='Perturbation',
#                       ymin=0, ymax=1, xscale='log', yscale='linear', 
#                       chance_classes=10, hline_lims=None):
#         ticks = self.ticks if self.ticks else ticks
#         if not ticks:
#             os.system('say "PLEASE SET THE TICKS YOU ABSOLUTE POTATO"')
#             raise Exception('Need to set ticks before plotting')
#         plot_accuracy_trajectory(self.accuracy[type], 
#                                  self.accuracy_baseline[type], xticks=ticks,
#                                  legend_loc=legend_loc, 
#                                  titleadd=f'{type} {titleadd}', 
#                                  ymin=ymin, ymax=ymax,
#                                  xscale=xscale, yscale=yscale, 
#                                  n_classes=chance_classes, 
#                                  hline_lims=hline_lims, xlabel=xlabel)
#         return


#     def plot_effective_dimensions(self, layer=1, ticks=None, xlabel='', 
#                                   titleadd='', legend_loc='lower right',
#                                   xlog=False, ylog=True, 
#                                   hline_lims = None, ylims=None):
#         ticks = self.ticks if self.ticks else ticks
#         if not ticks:
#             os.system('say "PLEASE SET THE TICKS YOU ABSOLUTE POTATO"')
#             raise Exception('Need to set ticks before plotting')

#         plot_effective_dimensions(self, ticks, xlabel, layer=layer, 
#                                   titleadd=titleadd, legend_loc=legend_loc, 
#                                   xlog=xlog, ylog=ylog, 
#                                   hline_lims=hline_lims, ylims=ylims)

#         return


class Perturbation:

    def __init__(self,
                 base_perturbations=default_perturbation_settings,
                 #batch_size=64,
                 dataset_class=datasets.CIFAR10,
                 experiment_settings=default_perturbation_settings, swap_list=None):#,
                 # columns=None, rows=None,
                 # val=(1, 1, 1),
                 # intensity=False,
                 # noise=False, var=None,
                 # random_pixels=None,
                 # classes=None,
                 # name=None):
        """
        Does not currently handle intensity modifications!

        :param base_perturbations: this is the dictionary of perturbation settings for the baseline model
        :param batch_size:
        :param dataset_class:
        :param columns: list of numbers of columns to perturb - can be adjusted over trials
        :param rows: list of numbers of rows to perturb - can be adjusted over trials
        :param val: the (mean) value to replace the pixel(s) with (R, G, B) float values in [-1, 1] - fixed over trials
        :param intensity: whether to clip based on intensity - fixed over trials
        :param noise: whether to vary the value according to a gaussian distribution - fixed over trials
        :param var: if noise, the variance of the distribution - can be adjusted over trials
        :param random_pixels: list of the number of random pixels to perturb
        """

        self.model_names = ('unperturbed', getattr(experiment_settings, 'name')) if getattr(experiment_settings, 'name') \
            else ('unperturbed', 'perturbed')

        # get the datasets
        # load in the baseline dataset
        self.baseline_settings = base_perturbations
        # set the seed
        set_seed(COMMON_SEED)
        self.loaders_baseline = subset_class_loader(base_perturbations)

        self.dataset_class = dataset_class

        self.batch_size = getattr(experiment_settings, 'batch_size')#batch_size

        self.n_classes = ((len(getattr(base_perturbations, 'unmod_classes')) if
                           getattr(base_perturbations, 'unmod_classes') else 0) +
                          (len(getattr(base_perturbations, 'mod_classes')) if getattr(base_perturbations, 'mod_classes')
                           else 0))

        # set the perturbation parameters
        self.columns = getattr(experiment_settings, 'columns')#columns
        self.rows = getattr(experiment_settings, 'rows')#rows
        self.random_pixels = getattr(experiment_settings, 'random_pixels') #random_pixels
        self.classes = getattr(experiment_settings, 'mod_classes') #classes

        self.val = getattr(experiment_settings, 'val') #val
        self.intensity = getattr(experiment_settings, 'intensity') #intensity
        self.noise = getattr(experiment_settings, 'noise')#noise
        self.var = getattr(experiment_settings, 'var')#var

        self.swaps = swap_list

        # get the number of trials from these
        lengths = [len(i) if i is not None else 0 for i in (self.columns, self.rows, self.random_pixels, self.var,
                                                            self.classes, self.swaps)]
        self.n_trials = max(lengths)
        # test to see if this is correct
        if not all((x == max(lengths) or x == 0) for x in lengths):
            os.system('say "Hey dingus, theres an error in your experiment setup"')
            raise Exception(f'Number of perturbations does not match between modes.'
                            f'Please ensure each list of values is of the same length: {lengths}')


        # results tracking
        self.accuracy_baseline = None
        self.accuracy_trials = {'train': [None]*self.n_trials,
                                'test': [None]*self.n_trials}

        self.dimensions_baseline = None
        self.dimensions_trials = []

        # distance and similarity tracking
        self.similarities = {'activations': [None] * self.n_trials,
                             'weights': [None] * self.n_trials}
        self.similarities_clipped = {'activations': [None] * self.n_trials,
                                     'weights': [None] * self.n_trials}
        self.distances = {'activations': [None] * self.n_trials,
                          'weights': [None] * self.n_trials}
        self.distances_clipped = {'activations': [None] * self.n_trials,
                                  'weights': [None] * self.n_trials}

        # distance and similarity quantities
        self.baseline_trace_clipped = {'activations': [None]*self.n_trials, 'weights': [None]*self.n_trials}
        self.baseline_trace = {'activations': [None]*self.n_trials, 'weights': [None]*self.n_trials}

        self.experiment_trace_clipped = {'activations': [None]*self.n_trials,
                                         'weights': [None]*self.n_trials}
        self.experiment_trace = {'activations': [None]*self.n_trials,
                                 'weights': [None]*self.n_trials}
        self.nuclear_norm_clipped = {'activations': [None]*self.n_trials,
                                     'weights': [None]*self.n_trials}
        self.nuclear_norm = {'activations': [None]*self.n_trials,
                             'weights': [None]*self.n_trials}

        # setting the tracking of the experiment objects
        self.experiment_models = list([None]*self.n_trials)

        # running parameters (set later)
        self.baseline_model = None
        self.n_neurons = None
        self.n_epochs = None

        self.layers = None

        return

    def set_baseline(self, n_epochs, n_neurons, w_clip_val=30):
        """
        Trains the reference model and experiment parameters to ensure accurate comparison
        :param n_epochs: number of epochs to train
        :param n_neurons: the list of the number of neurons in each hidden layer of the network
        :param w_clip_val: the rank at which to clip the weights
        :return:
        """
        # create the model
        set_seed(COMMON_SEED)  # set the seed
        baseline_model = spec.spectrum_analysis(n_neurons)
        self.n_neurons = n_neurons

        # train the model for n_epochs
        self.n_epochs = n_epochs # save so we can use the same number for later
        set_seed(COMMON_SEED) # set the seed
        baseline_train = self.loaders_baseline[0]
        baseline_val = self.loaders_baseline[1]
        display_dataloader_images(baseline_train, 8, display=True)

        # training
        #set_seed(COMMON_SEED) # set the seed
        baseline_model.train(baseline_train, baseline_val, n_epochs, 
                             grain=50000, ep_grain=n_epochs)

        # adding the accuracy to the record
        self.accuracy_baseline = baseline_model.val_history[-1]
        # get the weight covariances
        baseline_model.get_weight_covs()
        # and effective dimensions
        baseline_model.get_effective_dimensions(clip=w_clip_val)
        self.dimensions_baseline = [baseline_model.effective_dimensions[j][-1] for j in range(len(n_neurons))]

        self.w_clip_val = w_clip_val
        self.baseline_model = baseline_model
        return

    def run_perturbation_training(self, layers, perturbation_level, w_clip_val=None, a_clip_val=64, plot=True,
                                  classes=tuple(range(10))):
        """
        Runs one iteration of the pertrbation experiement per the settings initialized upon object creation. Updates the
        result attributes.

        :param layers:              list(int)   :  which layers to perform analysis on
        :param perturbation_level:  int         : which level of perturbation to run
        :param w_clip_val:          int         : the rank at which to clip the weights for comparison. By default, this
                                                 is the rank set during the training of the baseline
        :param a_clip_val:          int         : the rank at which to clip the activations for comparison
        :param plot:                bool        : whether to plot and save the eigenvector similarity plots
        :param classes:             list(int)   : the classes to include (and perturb) in the training of the network
        :return:
        """

        # check perturbation level
        if perturbation_level > self.n_trials:
            os.system('say "set a valid perturbation level you absolute chode"')
            raise Exception('Perturbation level must be less than the number of trials')

        w_clip_val = w_clip_val if w_clip_val else self.w_clip_val

        start = time.time()
        # for each trial
        j = perturbation_level
        #for j in tqdm(range(trial_start, trial_end), desc='running the perturbations'):
        # first, set up the indices if there are rows and or columns
        j_col = self.columns[j] if self.columns else None
        j_row = self.rows[j] if self.rows else None
        j_pix = self.random_pixels[j] if self.random_pixels else None
        j_var = self.var[j] if self.var else None
        j_cls = self.classes[j] if self.classes else self.baseline_settings.unmod_classes
        j_swp = self.swaps[j] if self.swaps else None

        #print(j_col, j_row, j_pix, j_var, j_cls, j_swp)

        if j_cls:
            if classes:
                print('Warning: Number of classes was selected by experiment set-up, not the given parameter. To avoid '
                      ' this warning in the future, set input classes to None.')
            classes = j_cls

        #print(classes)

        class_loader_settings = PerturbationSettings(columns=j_col, rows=j_row, val=self.val, intensity=self.intensity,
                                                     var=j_var, noise=self.noise, mod_classes=classes,
                                                     random_pixels=j_pix, unmod_classes=[])

        # create the dataloader
        set_seed(COMMON_SEED)  # set the seed
        trial_loaders = subset_class_loader(class_loader_settings, swap=j_swp)

        # show the dataset
        display_dataloader_images(trial_loaders[0], 8, display=True)

        # train the model
        set_seed(COMMON_SEED)  # set the seed
        trial_model = spec.spectrum_analysis(self.n_neurons) # create the model
        set_seed(COMMON_SEED)  # set the seed
        trial_model.train(trial_loaders[0], trial_loaders[1], #self.n_epochs, grain=8, ep_grain=2)#
                          self.n_epochs, grain=50000, ep_grain=self.n_epochs) # train the model
        trial_model.get_effective_dimensions(clip=w_clip_val)  # get the effective dimensions
        # seeing what is up with the spectrum on the noise
        if plot:
            trial_model.plot(plotlist=['rel_eds', 'rel'], saveadd=f'_p--{perturbation_level}')


        self.accuracy_trials['test'][j] = trial_model.val_history[-1]   # add the accuracy results
        self.accuracy_trials['train'][j] = trial_model.train_history[-1]    # add the accuracy results

        self.dimensions_trials.append([trial_model.effective_dimensions[i][-1] for i in range(len(self.n_neurons))])
        # add the effective dimensions

        # create the similarity object
        simobj = sim.network_comparison(self.baseline_model, trial_model, names=self.model_names)

        # do the similarity analysis
        # get the alignments
        simobj.compute_alignments(self.loaders_baseline[0], layers) # compute alignments for each layer
        simobj.compute_cossim()

        # get the metrics and add them to the correct list
        ###### DISTANCES
        ## clipped values
        act, way, quants = simobj.network_distance(w_clip=w_clip_val, a_clip=a_clip_val, return_quantities=True)    # these come back with all layers
        self.distances_clipped['activations'][j] = [act[ind-1] for ind in layers]
        self.distances_clipped['weights'][j] = [way[ind-1] for ind in layers]
        # convert the component quantities to get the layer lists for this perturbation
        trbasew, trexp_w, nnorm_w = convert_dist_component_dict(quants, 'weights')
        trbasea, trexp_a, nnorm_a = convert_dist_component_dict(quants, 'activations')
        self.experiment_trace_clipped['weights'][j] = [trexp_w[ind-1] for ind in layers]
        self.experiment_trace_clipped['activations'][j] = [trexp_a[ind - 1] for ind in layers]
        self.nuclear_norm_clipped['weights'][j] = [nnorm_w[ind-1] for ind in layers]
        self.nuclear_norm_clipped['activations'][j] = [nnorm_a[ind - 1] for ind in layers]

        self.baseline_trace_clipped['weights'][j] = [trbasew[ind-1] for ind in layers]
        self.baseline_trace_clipped['activations'][j] = [trbasea[ind-1] for ind in layers]

        ## unlcipped values
        act, way,  quants = simobj.network_distance(w_clip=None, return_quantities=True)
        self.distances['activations'][j] = [act[ind-1] for ind in layers]
        self.distances['weights'][j] = [way[ind-1] for ind in layers]
        # convert the component quantities to get the layer lists for this perturbation
        trbasew, trexp_w, nnorm_w = convert_dist_component_dict(quants, 'weights')
        trbasea, trexp_a, nnorm_a = convert_dist_component_dict(quants, 'activations')
        self.experiment_trace['weights'][j] = [trexp_w[ind - 1] for ind in layers]
        self.experiment_trace['activations'][j] = [trexp_a[ind - 1] for ind in layers]
        self.nuclear_norm['weights'][j] = [nnorm_w[ind - 1] for ind in layers]
        self.nuclear_norm['activations'][j] = [nnorm_a[ind - 1] for ind in layers]

        self.baseline_trace['weights'][j] = [trbasew[ind-1] for ind in layers]
        self.baseline_trace['activations'][j] = [trbasea[ind-1] for ind in layers]

        ###### SIMILARITIES
        ## clipped values
        act, way = simobj.network_distance(w_clip=w_clip_val, a_clip=a_clip_val,
                                           sim=True, return_quantities=False) # we already got the quantities from above
        self.similarities_clipped['activations'][j] = [act[ind-1] for ind in layers]
        self.similarities_clipped['weights'][j] = [way[ind-1] for ind in layers]
        ## unclipped values
        act, way = simobj.network_distance(w_clip=None, sim=True, return_quantities=False)# we already got the quantities from above
        self.similarities['activations'][j] = [act[ind-1] for ind in layers]
        self.similarities['weights'][j] = [way[ind-1] for ind in layers]

        if plot:
            simobj.plot_sims(quantities=['activations'], clips=[int(a_clip_val*1.5)]*len(self.n_neurons),
                             filename_append=f'p--{j}',
                             plot_clip=a_clip_val)
            simobj.plot_sims(quantities=['weights'], clips=[w_clip_val+20]*len(self.n_neurons),
                             filename_append=f'p--{j}',
                             plot_clip=w_clip_val)

        self.layers = layers

        # save to the list
        self.experiment_models[j] = trial_model
        #del trial_model
        #del simobj
        print(f'TOTAL TIME: {(time.time() - start):.2f} seconds')
        return

    def reshape_results(self):
        """
        reshapes results from how they are stored into a more intuitive form for analysis
        measurement -> layer -> quantity -> values in perturbation order
        """
        self.similarities = reshape_multilayer_results_dict(self.similarities, self.layers)
        self.similarities_clipped = reshape_multilayer_results_dict(self.similarities_clipped, self.layers)

        self.distances = reshape_multilayer_results_dict(self.distances, self.layers)
        self.distances_clipped = reshape_multilayer_results_dict(self.distances_clipped, self.layers)

        # also reshape the components
        self.experiment_trace = reshape_multilayer_results_dict(self.experiment_trace, self.layers)
        self.experiment_trace_clipped = reshape_multilayer_results_dict(self.experiment_trace_clipped, self.layers)
        self.nuclear_norm = reshape_multilayer_results_dict(self.nuclear_norm, self.layers)
        self.nuclear_norm_clipped = reshape_multilayer_results_dict(self.nuclear_norm_clipped, self.layers)

        self.baseline_trace = reshape_multilayer_results_dict(self.baseline_trace, self.layers)
        self.baseline_trace_clipped = reshape_multilayer_results_dict(self.baseline_trace_clipped, self.layers)

        self.dimensions_trials = reshape_multilayer_dimensions(self.dimensions_trials, self.layers)
        print('Reshaped results for multilayer format')
        return

    # def plot_trajectories(self, layers=None,
    #                       wsim_order=(2, 4, 3, 1, 0),
    #                       asim_order=(2, 3, 0, 1),
    #                       wdist_order=(3, 0, 1, 4, 2),
    #                       adist_order=(0, 1, 3, 2),
    #                       ws_loc='upper left',
    #                       as_loc='upper left',
    #                       wd_loc='lower right',
    #                       ad_loc='upper left',
    #                       xticks =None,
    #                       xlabadd =None,
    #                       ylabadd =None,
    #                       titleadd ='',
    #                       ylog=True):
    #     """

    #     :param layers: list(int)    : the layers to plot the trajectories for
    #     :param wsim_order:
    #     :param asim_order:
    #     :param wdist_order:
    #     :param adist_order:
    #     :param ws_loc:
    #     :param as_loc:
    #     :param wd_loc:
    #     :param ad_loc:
    #     :param xticks:
    #     :param xlabadd:
    #     :param ylabadd:
    #     :param titleadd:
    #     :param ylog:
    #     :return:
    #     """
    #     layers = layers if layers else self.layers

    #     for layer in layers:
    #         print(f'Plotting results for layer {layer}')
    #         plot_result_trajectories(self.similarities[layer],
    #                                  self.similarities_clipped[layer],
    #                                  self.distances[layer],
    #                                  self.distances_clipped[layer],
    #                                  wsim_order=wsim_order, asim_order=asim_order,
    #                                  wdist_order=wdist_order, adist_order=adist_order,
    #                                  ws_loc=ws_loc, as_loc=as_loc, wd_loc=wd_loc, ad_loc=ad_loc,
    #                                  xticks=xticks, xlabadd=xlabadd, ylabadd=ylabadd,
    #                                  titleadd = f'{titleadd} - Layer {layer}',
    #                                  ylog=ylog)
    #     return

    def plot_accuracy(self, xticks=None, legend_loc = 'lower left', titleadd=''):
        plot_accuracy_trajectory(self.accuracy_trials, self.accuracy_baseline, xticks=xticks,
                                 legend_loc=legend_loc, titleadd=titleadd)

        return

    def save_trajectories(self, pickle_name, description='', save=True):
        """
        Creates a results dictionary which can be used to create a results object or saved to a pickle object for later
        access.
        :param pickle_name: str : the name of the dictionary to be saved
        :param description: str : a description of the experiment for later access
        :param save: bool       : whether to save as a pickled variable
        :return: results : dict : the results dictionary
        """
        names = ['sims_u', 'sims_c', 'dist_u', 'dist_c', 'acc', 'acc_base', 'effdims', 'base_dims',
                 'base_tr_u', 'base_tr_c', 'exp_tr_u', 'exp_tr_c', 'nnorm_u', 'nnorm_c', 'models']
        dicts = [self.similarities, self.similarities_clipped,
                 self.distances, self.distances_clipped,
                 self.accuracy_trials, {'test': self.baseline_model.val_history[-1],
                                        'train': self.baseline_model.train_history[-1]},
                 self.dimensions_trials, self.dimensions_baseline,
                 self.baseline_trace, self.baseline_trace_clipped,
                 self.experiment_trace, self.experiment_trace_clipped,
                 self.nuclear_norm, self.nuclear_norm_clipped,
                 {'base': self.baseline_model, 'experiment': self.experiment_models}]
        results = {}

        # create the results dictionary
        results = dict(zip(names, dicts))
        #for i in range(len(names)):
        #    results[names[i]] = dicts[i]

        results['README'] = description

        if save:
            # save the results
            PATH = f'../pickle_vars/{pickle_name}.pkl'
            with open(PATH, 'wb') as f:
                pickle.dump(results, f)

        return results

##############################
###### Helper Functions ######
##############################

########################################################### Displaying images
# show the images
# for showing the image
def imshow(img, ticks=[[],[]]):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xticks(ticks[0], labels=ticks[1])
    plt.yticks([])
    plt.show()


def display_dataloader_images(dataloader, n_images, display=False):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    #if display:
    #    ticks = ([32*k for k in range(1, n_images+1)], [class_names[labels[j]] for j in range(n_images)])
    #else:
    ticks = [[],[]]

    imshow(torchvision.utils.make_grid(images[0:n_images]), ticks=ticks)

    if display:
        print([class_names[labels[j]] for j in range(n_images)])

    return

def display_sorted_class_images(dataloader, n_images, class_ind=0):
    dataiter = iter(dataloader)

    # get the first n_images instances where class ind = desired
    k = 0
    image_list = []
    while k < n_images:
        images, labels = next(dataiter)
        
        i = 0 
        for label in labels:
            if label == class_ind and k < n_images:
                image_list.append(images[i])
                k += 1
            i += 1
    
    # show the images
    imshow(torchvision.utils.make_grid(image_list))

    return




############################################################# Generating Datasets
# for generating the datasets
# def get_first_n_from_class(dataset, count, class_ind, random=False):
    """
    Sampling without replacement
    """
#     if random:
#         all_class_inds = [i for i, (e,c) in enumerate(dataset) if c == class_ind]
#         # random order and choose first n
#         mind_list = list(np.random.permutation(all_class_inds))[:count]
        
#     else:
#         mind_list = [i for i, (e,c) in enumerate(dataset) if c == class_ind][:count]
#     #print(mind_list)
#     #print(f'Number of indices for class {class_ind} : {len(mind_list)}')
#     return mind_list

def get_first_n_from_class(dataset, count, class_ind, random=None):
    """
    Sampling with replacement 
    random still parameter for compatibility just switching between these two
    functions if desired. But for this function, it serves no purpose :)
    """
    all_class_inds = [i for i, (e,c) in enumerate(dataset) if c == class_ind]
    
    # get the samples with replacement
    sampler = np.random.default_rng()
    mind_list = list(sampler.choice(all_class_inds, size=count, 
                     replace=True))

    return mind_list

def get_first_n_from_class_lists(dataset, counts, class_inds, random=False):
    assert len(counts)==len(class_inds), \
           'Counts list and index list should be the same length'

    index_list = []
    for k in range(len(counts)):
        index_list += get_first_n_from_class(dataset, counts[k], class_inds[k],
                                             random=random)
    
    return index_list
        


def subset_class_loader(subset_settings : PerturbationSettings = default_perturbation_settings, swap=None):

    """
    :param class_indices: the indices of unperturbed classes
    :param batch_size: batch size for the dataloader. Default value is 64
    :param dataset_class:
    :param mod_ind: the indices of the perturbed classes
    :param columns: which columns to perturb
    :param rows: which rows to perturb
    :param val: what value to set each of the perturbed columns or rows to, or the max intensity before clipping
    :param intensity: whether or not to clip in intensity space
    :param noise: whether or not to add gaussian noise to the modified pixels
    :param var: the variance of the gaussian noise if noise is true
    :param random_pixels: alternatively, modify random pixels.
    :return: train loader and val loader according to these specifications
    """
    class_indices = getattr(subset_settings, 'unmod_classes')
    mod_ind = getattr(subset_settings, 'mod_classes')
    columns = getattr(subset_settings, 'columns')
    rows = getattr(subset_settings, 'rows')
    val = getattr(subset_settings, 'val')
    intensity = getattr(subset_settings, 'intensity')
    noise = getattr(subset_settings, 'noise')
    var = getattr(subset_settings, 'var')
    random_pixels = getattr(subset_settings, 'random_pixels')

    dataset_class = getattr(subset_settings, 'dataset_class')
    #print(dataset_class)
    batch_size = getattr(subset_settings, 'batch_size')

    mod_class_count = getattr(subset_settings, 'class_count')
    mod_random_subsets = getattr(subset_settings, 'random_subsets')
    resize = getattr(subset_settings, 'resize')
    greyscale = getattr(subset_settings, 'greyscale')

    # get unmodified classes first
    # load the entire dataset
    trainset, valset = nn_mod.get_datasets(dataset_class=dataset_class)

    # get just the subset
    indices_train = [i for i, (e, c) in enumerate(trainset) if c in class_indices]
    indices_val = [i for i, (e, c) in enumerate(valset) if c in class_indices]

    # get the subset
    trainset_sub = torch.utils.data.Subset(trainset, indices_train)
    valset_sub = torch.utils.data.Subset(valset, indices_val)

    #print(len(trainset_sub), len(valset_sub))

    # setting up the mod transform if it exists
    #mod_trans = transforms.Compose(transforms.Pad(padding=8))

    # if mod class, then mod class
    if mod_ind:
        # get the indices
        if not mod_class_count:
            mind_train = [i for i, (e, c) in enumerate(trainset) if c in mod_ind]
        elif mod_random_subsets:
            mind_train = get_first_n_from_class_lists(trainset, mod_class_count, 
                                                mod_ind, random=True)
        else:
            mind_train = get_first_n_from_class_lists(trainset, mod_class_count, mod_ind)

        print(f'number of images in training data: ', len(mind_train))
        #print(f'Indices : {mind_train}')
        mind_val = [i for i, (e, c) in enumerate(valset) if c in mod_ind]
    
        # get the subsets
        m_train_sub = torch.utils.data.Subset(trainset, mind_train)
        m_val_sub = torch.utils.data.Subset(valset, mind_val)
        #print(len(m_train_sub), len(m_val_sub))

        if swap:
            # getting the indices is fine
            if swap[0].old_outside:
                # fix this later to check all of them but for now OK
                mod_ind_news = [s.new_label for s in swap]
                mod_ind_olds = [s.old_label for s in swap]
                mod_ind_adds = []
                for i in mod_ind:
                    if (i not in mod_ind_news) and (i not in mod_ind_olds):
                        mod_ind_adds.append(i)

                mod_ind_news += mod_ind_adds

                mod_ind_use = []
                for i in range(len(swap)):
                    mods_single = mod_ind_news.copy()
                    mods_news_need = [swap[j].old_label for j in range(i, len(swap))]
                    #mods_single.append(s.old_label)
                    mod_ind_use.append(mods_single + mods_news_need)
            else:
                mod_ind_use = list(mod_ind) * len(swap)
            #print(mod_ind_use)

            # this is where the problem happens for some reason
            # when done in sequence
            for i in range(len(swap)):
                setting = swap[i]
                #print(f'mod_ind_use = {mod_ind_use[i]}')
                #print(setting.__dict__)

                m_train_sub, m_val_sub = swap_trainset_labels(setting, mod_ind_use[i], m_train_sub, m_val_sub)
                if i < len(swap) - 1:
                    m_train_sub, m_val_sub = m_train_sub.dataset, m_val_sub.dataset


            #m_train_sub, m_val_sub = swap_trainset_labels(swap, mod_ind, m_train_sub, m_val_sub)
        transform_list = [dataset_perturbations(column_indices=columns,
                                                row_indices=rows,
                                                val=val,
                                                intensity=intensity,
                                                noise=noise, var=var,
                                                random_pixels=random_pixels)]
        if resize:
            transform_list.append(torchvision.transforms.Resize(resize))
        if greyscale:
            transform_list.append(torchvision.transforms.Grayscale())

        if dataset_class == datasets.CIFAR10:
            mod_transform = transforms.Compose(transform_list)
        if dataset_class == datasets.MNIST:
            mod_transform = transforms.Compose(transform_list)

        modded_train = MyDataset(m_train_sub, transform=mod_transform)
        modded_val = MyDataset(m_val_sub, transform=mod_transform)

        trainset_sub = torch.utils.data.ConcatDataset([trainset_sub, modded_train])
        valset_sub = torch.utils.data.ConcatDataset([valset_sub, modded_val])

    # get the dataloader
    train_loader_sub = nn_mod.get_dataloader(batch_size, trainset_sub, shuffle=False)
    val_loader_sub = nn_mod.get_dataloader(batch_size, valset_sub, shuffle=False)

    return train_loader_sub, val_loader_sub

# Swapping images between classes
def swap_trainset_labels(swap_settings : SwapSettings, train_classes, trainset_subset : torch.utils.data.Subset,
                         valset_subset : torch.utils.data.Subset):
    if swap_settings.bidirectional:
        j = 0

    k = 0
    for ind, target in enumerate(trainset_subset.dataset.targets):
        if (target == swap_settings.old_label and k < swap_settings.n_images):
            trainset_subset.dataset.targets[ind] = swap_settings.new_label
            k += 1
        if swap_settings.bidirectional: # if bidirectional, also be checking to swap the other images
            if (target == swap_settings.new_label and j < swap_settings.n_images):
                trainset_subset.dataset.targets[ind] = swap_settings.old_label
                j += 1

        if (not swap_settings.bidirectional) and (k >= swap_settings.n_images):
            break
        elif (k >= swap_settings.n_images and j >= swap_settings.n_images):
            break

    # if the original label of the images being relabelled is outside of the training set
    if swap_settings.old_outside:
        train_classes = [k for k in train_classes if k != swap_settings.old_label]
        print(f'Classes used for Training with swapped labels: {train_classes}')

        indices_train_again = [i for i, (e, c) in enumerate(trainset_subset) if c in train_classes]
        #print('train len indices ', len(indices_train_again))
        indices_val_again = [i for i, (e, c) in enumerate(valset_subset) if c in train_classes]
        #print('test len indices ', len(indices_val_again))
        # get the subset
        trainset_subset = torch.utils.data.Subset(trainset_subset, indices_train_again)
        valset_subset = torch.utils.data.Subset(valset_subset, indices_val_again)

        #print('train len subset ', len(trainset_subset))


    return trainset_subset, valset_subset


# custom dataset to apply transforms to
class MyDataset:
    """
    Custom dataset for applying various perturbations
    """
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


# dataset purturbations class for the transformations
class dataset_perturbations(object):
    def __init__(self, val=(1, 1, 1),
                 column_indices=None, row_indices=None,
                 intensity=False,
                 noise=False, var=0,
                 random_pixels=None):
        '''

        :param val:
        :param column_indices:
        :param row_indices:
        :param intensity:
        :param noise:
        :param var:
        :param random_pixels:
        '''
        #set_seed(COMMON_SEED)
        self.val = val
        self.columns = column_indices
        self.rows = row_indices
        self.intensity = intensity

        self.random_pix = random_pixels

        if noise:
            #if var == 0:
            #    os.system('say "Hey dingus, change the first variance to None for no perturbation"')
            #    raise Exception('If using random masking, please include nonzero variance')
            # if self.columns or self.rows:
            #     self.random = scipy.stats.norm(0, np.sqrt(var))
            #     self.std = np.sqrt(var)
            # else:
                self.random = True
                self.std = np.sqrt(var) if var else 0
        else:
            self.random = None
            self.std = None


    def __call__(self, img_tensor):  # this should always come after the toTensor transform
        """

        :param img_tensor: image in the form of a tensor
        :param val:
        :param column_indices:
        :param row_indices:
        :return:
        """
        # def normalize_tensor(im_tensor):
        #     # normalize from [-1, 1] into [0,1] without clipping
        #     im_tensor = im_tensor/2 + torch.ones_like(im_tensor)
        #     return im_tensor
        #
        # img_tensor = normalize_tensor(img_tensor)
        #print(torch.max(img_tensor) - torch.min(img_tensor))


        #print(img_tensor.size())
        # Do some transformations. Here, we're just passing though the input
        #set_seed(COMMON_SEED)
        if self.columns:
            for index in self.columns:
                img_tensor[0, :, index] = self.val[0]
                if img_tensor.shape[0] > 1:
                    img_tensor[1, :, index] = self.val[1]
                    img_tensor[2, :, index] = self.val[2]

            if self.random and self.std > 0:
                for index in self.columns:
                    #set_seed(COMMON_SEED)
                    img_tensor[:, :, index] += torch.randn(img_tensor[:, :, index].size()) * self.std

        if self.rows:
            for index in self.rows:
                img_tensor[0, index, :] = self.val[0]
                if img_tensor.shape[0] > 1:
                    img_tensor[1, index, :] = self.val[1]
                    img_tensor[2, index, :] = self.val[2]

            if self.random and self.std > 0:
                for index in self.columns:
                    #set_seed(COMMON_SEED)
                    img_tensor[:, index, :] += torch.randn(img_tensor[:, index, :].size()) * self.std

        # if self.random_pix:
        #     #shape = img_tensor.shape
        #     #indices_long = np.random.choice(np.arange(shape[1]* shape[2]), size=self.random_pix)
        #     #indices = [divmod(indices_long[j], shape[1]) for j in range(self.random_pix)]
        #
        #     for index in self.random_pix:
        #         #add_val = torch.tensor(self.val) + self.random.rvs(size=3) if self.random else torch.tensor(self.val)
        #         img_tensor[:, index[0], index[1]] = torch.tensor(self.val)
        #
        #     if self.random:
        #         for index in self.random_pix:
        #             img_tensor[:, index[0], index[1]] += self.random.rvs(size=3)

        if self.random and not (self.rows or self.columns) and self.std > 0:
            #set_seed(COMMON_SEED)
            img_tensor += torch.randn(img_tensor.size()) * self.std

        if self.intensity:
            img_tensor = torch.clamp(img_tensor, min=0, max=self.val)

        # normalize to be in the range [0,1]

        return img_tensor

    def __repr__(self):
        return "Turning Columns and or Rows a Color"


def get_random_pixel_indices(img_shape, total_pix):
    indices_long = np.random.choice(np.arange(img_shape[0]*img_shape[1]), size=total_pix)
    indices = [divmod(indices_long[j], img_shape[1]) for j in range(total_pix)]

    return indices


################################################################# Plotting Results
# Plotting Functions
# def plot_result_trajectories(similarities, similarities_clipped, distances, distances_clipped,
#                                     xticks=None,
#                                     xlabadd=None,
#                                     ylog=True, xlog=False, plot_baselines=True,
#                                     title=None, ylims=None):
#     xlabadd = xlabadd if xlabadd else 'Perturbation'

#     # create the figure
#     fig, axs = plt.subplots(2, 2, sharex='col', figsize=(15, 8))
#     quant = {0: 'activations', 1: 'weights'}
#     y_labels = {0: 'absolute cosine\nsimilarity', 1: 'distance'}
#     metric = {0: (similarities, similarities_clipped), 1: (distances, distances_clipped)}

#     baselines = {'weights' : {'absolute cosine\nsimilarity': (way_id_sim, 'wsu', 'wsc'),
#                               'distance' : (way_id_dist, 'wdu', 'wdc')},
#                     'activations' : {'absolute cosine\nsimilarity' : (act_id_sim, 'as'),
#                                      'distance' : (act_id_dist, 'ad')}}


#     for i in (0, 1):
#         for j in (0, 1):
#             ax = axs[i, j]

#             # quant is j, metric is i
#             # ( i is rows, j is column )
#             if ylims and i == 1:
#                 if ylims[quant[j]]:
#                     limits = ylims[quant[j]]
#                     ax.set_ylim(limits[0], limits[1])

#             #### setting labels and scales
#             if j == 0:
#                 ax.set_ylabel(y_labels[i], fontsize=axis_fontsize)
#             if i == 1:
#                 ax.set_xlabel(xlabadd, fontsize=axis_fontsize)
#                 if j == 0 and ylog:
#                     ax.set_yscale('log')

#             if i == 0:
#                 ax.set_ylim(0, 1.05)

#                 if j == 0:
#                     ax.set_title('Activations', fontsize=axis_fontsize)
#                 else:
#                     ax.set_title('Weights', fontsize=axis_fontsize)


#             # actually plotting
#             xu = np.array(metric[i][0][quant[j]])
#             tick_places = np.array(range(len(xu))) if not xticks else np.array(xticks)
#             if plot_baselines:
#                 # get all the baselines
#                 bases = baselines[quant[j]][y_labels[i]]
#                 # identical first (2)
#                 ax.hlines(bases[0], min(tick_places), max(tick_places), colors='orange', linestyles=':',
#                            label='identical init')
#                 # then unclipped (3)
#                 ax.hlines(np.mean(quantities[bases[1]][2]), min(tick_places), max(tick_places), colors='r',
#                            linestyles=':', label='random init (unclipped)')
#                 # then if the clipped exists (4)
#                 if len(bases) == 3:
#                     ax.hlines(np.mean(quantities[bases[2]][2]), min(tick_places), max(tick_places), colors='b',
#                                linestyles=':', label='random init (clipped)')

#             # unclipped first (0)
#             xu = np.array(metric[i][0][quant[j]])
#             tick_places = np.array(range(len(xu))) if not xticks else np.array(xticks)
#             mask = np.isfinite(xu)
#             ax.scatter(tick_places[mask], xu[mask], marker='o', color='r', label='unclipped')

#             # clipped (1)
#             xc = np.array(metric[i][1][quant[j]])
#             mask = np.isfinite(xc)
#             ax.scatter(tick_places[mask], xc[mask], marker='o', color='b', label='clipped')
#             ax.scatter(tick_places[mask], xc[mask], color=colors.to_rgba('white', 0), label= '')

#             # do the custom legend order
#             if i == 0 and j == 1:
#                 # (lines, labels) = plt.gca().get_legend_handles_labels()
#                 # lines.insert(1, plt.Line2D(tick_places, xc[mask], linestyle='none', marker='none'))
#                 # labels.insert(1, '')
#                 # make the legend when everything is there
#                 ax.legend(loc='upper center', bbox_to_anchor=(0, -1.25), fontsize=axis_fontsize, numpoints=1, ncol=2)

#             if xlog:
#                 ax.set_xscale('log')

#             ax.tick_params(axis='both', which='both', labelsize=axis_fontsize)

#     plt.subplots_adjust(hspace=0, wspace=0.1)
#     if title:
#         plt.suptitle(title, fontsize=title_fontsize)
#     #plt.tight_layout()
#     plt.show()
#     return


def plot_accuracy_trajectory(accuracies, acc_baseline, xticks=None, legend_loc='lower left', titleadd='', 
                             xlabel='Perturbation',
                             ymin=0.0, ymax=1, xscale='log', yscale='linear', n_classes=10, hline_lims=None):
    fig = plt.figure(figsize=(10,5))
    xticks = xticks if xticks else list(range(len(accuracies)))
    print(xticks)

    hline_lims = hline_lims if hline_lims else (min(xticks), max(xticks))

    plt.plot(xticks, accuracies, label='perturbed models', color='m', marker='o', linewidth=0)
    plt.hlines(acc_baseline, hline_lims[0], hline_lims[1], label='unperturbed', color='g')
    plt.hlines(1 / n_classes, hline_lims[0], hline_lims[1], label='chance', color='orange', linestyle=':')


    plt.legend(loc=legend_loc, fontsize=axis_fontsize)

    plt.ylim(ymin, ymax)

    plt.xscale(xscale)
    plt.yscale(yscale)

    plt.xlabel(xlabel, fontsize=axis_fontsize)
    plt.ylabel(f'{titleadd} performance', fontsize=axis_fontsize)
    #plt.title(f'Accuracy Trajectory {titleadd}', fontsize=title_fontsize)
    plt.tick_params(axis='both', which='both', labelsize=axis_fontsize)
    plt.show()

    return


default_ranges = {'weights': ((5e-3, 50), (0.005, 1)),
                  'activations': ((1e2, 1e7), (0.005, 1))}

def plot_trace_nnorms(results_holder, ticks, layer=1, clipped=True, titleadd='', xlabel='Perturbation Level',
                             legend_loc='best', yrange_weights=None, yrange_activations=None,
                             xlog=False, ylog=True, hline_lims=None):

    yrange_weights = yrange_weights if yrange_weights else default_ranges['weights'][0]
    yrange_activations = yrange_activations if yrange_activations else default_ranges['activations'][0]
    hline_lims = hline_lims if hline_lims else (min(ticks), max(ticks))

    # check to see it is proper length
    if (len(yrange_weights) != 2) or (len(yrange_activations) != 2):
        os.system('say "Hey dingus, I cant set an axis limit that is not values! Idiot"')
        raise Exception('Please ensure there are exactly two values for each of the axis limit settings. This can be'
                        'as either a list or as a tuple.')

    trial = results_holder
    # get the sequences for the activations
    distance_activations = trial.distances_clipped[layer]['activations'] if clipped else (
        trial.distances)[layer]['activations']
    exp_trace_activations = trial.experiment_trace_clipped[layer]['activations'] if clipped \
        else trial.experiment_trace[layer]['activations']
    base_trace_activations = trial.baseline_trace_clipped[layer]['activations'][0] if clipped \
        else trial.baseline_trace[layer]['activations'][0]
    nuc_norm_activations = trial.nuclear_norm_clipped[layer]['activations'] if clipped else (
        trial.nuclear_norm)[layer]['activations']

    # get the sequences for the weights
    distance_weights = trial.distances_clipped[layer]['weights'] if clipped else trial.distances[layer]['weights']
    exp_trace_weights = trial.experiment_trace_clipped[layer]['weights'] if clipped else (
        trial.experiment_trace)[layer]['weights']
    base_trace_weights = trial.baseline_trace_clipped[layer]['weights'][0] if clipped else (
        trial.baseline_trace)[layer]['weights'][0]
    nuc_norm_weights = trial.nuclear_norm_clipped[layer]['weights'] if clipped else trial.nuclear_norm[layer]['weights']

    # quants
    distances = (distance_activations, distance_weights)
    exp_traces = (exp_trace_activations, exp_trace_weights)
    base_traces = (base_trace_activations, base_trace_weights)
    nuc_norms = (nuc_norm_activations, nuc_norm_weights)

    # setting the color for the measure
    meas_c = 'b' if clipped else 'r'

    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    titles = {0: 'Activations', 1: 'Weights'}

    for i in range(2):
        # 0 is activations, 1 is weights
        ax = axs[i]
        ax.set_xlabel(xlabel, fontsize=axis_fontsize)

        if i == 0:
            ax.set_ylabel(f'Component Quantities', fontsize=axis_fontsize)
            ax.set_ylim(yrange_activations[0], yrange_activations[1])
        if i == 1:
            ax.set_ylim(yrange_weights[0], yrange_weights[1])

        # plot the distance
        ax.plot(ticks, distances[i], color=meas_c, label=(r'distance = $Tr(C_{ref}) + Tr(C_{exp}) - 2\cdot '
                                                          r'||C_{ref}^{1/2} C_{exp}^{1/2}||_{nuc}$'), marker='o')
        # plt exp trace
        ax.plot(ticks, exp_traces[i], label=r'$Tr(C_{exp})$', color='m', marker='^', linestyle=':')
        # plot baseline trace
        ax.hlines(base_traces[i], hline_lims[0], hline_lims[1], label=r'$Tr(C_{ref})$', colors='k', linestyles=':')
        # plot nuc norm
        ax.plot(ticks, nuc_norms[i], color='orange', label=r'$||C_{ref}^{1/2} C_{exp}^{1/2}||_{nuc}$',
                marker='^', linestyle=':')
        # set the title
        ax.set_title(titles[i], fontsize=axis_fontsize)
        ax.tick_params(axis='both', which='both', labelsize=axis_fontsize)
        # set the scale
        if ylog:
            ax.set_yscale('log')
        if xlog:
            ax.set_xscale('log')

        if i == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0.2, -0.15), ncol=2, fontsize=axis_fontsize)

    cliptit = 'clipped' if clipped else 'unclipped'
    plt.subplots_adjust(wspace=0.15)
    plt.suptitle(f'Layer {layer} ({cliptit})', fontsize=title_fontsize)

    plt.show()
    return


def plot_effective_dimensions(results_holder, ticks, xlabel, layer=1, titleadd='', legend_loc='best', hline_lims = None,
                              xlog=False, ylog=True, ylims=None):
    fig = plt.figure(figsize=(10, 5))

    hline_lims = hline_lims if hline_lims else (min(ticks), max(ticks))
    plt.plot(ticks, results_holder.dimensions_trials[layer],
             label='Perturbed', marker='o', color='m')
    plt.hlines(results_holder.dimensions_baseline[layer-1], hline_lims[0], hline_lims[1], label='Baseline', 
               color='green')

    #plt.xticks(range(len(ticks)), ticks)
    #plt.xlim(-0.5, max)
    plt.xlabel(xlabel, fontsize=axis_fontsize)
    if xlog:
        plt.xscale('log')
    plt.ylabel('effective dimensions\n(expected rank)', fontsize=axis_fontsize)
    if ylog:
        plt.yscale('log')

    if ylims:
        plt.ylim(ylims[0], ylims[1])
    #plt.title(f'Effective Dimensions{titleadd}', fontsize=title_fontsize)
    plt.title(f'Layer {layer}', fontsize=axis_fontsize)
    plt.legend(loc=legend_loc, fontsize=axis_fontsize)
    plt.tick_params(axis='both', which='both', labelsize=axis_fontsize)

    plt.show()
    return


def plot_metric_vs_accuracy(results_obj, metric='similarity', layer=1, quantity='weights', experiment_name='', 
                            legend_loc='best', xlog=False, ylog=False):

    # get the measurements to plot
    if metric == 'similarity':
        clipped = getattr(results_obj, 'similarities_clipped')[layer][quantity]
        unclipped = getattr(results_obj, 'similarities')[layer][quantity]
    elif metric == 'distance':
        clipped = getattr(results_obj, 'distances_clipped')[layer][quantity]
        unclipped = getattr(results_obj, 'distances')[layer][quantity]
    else:
        os.system('say "You absolute bufoon, you didn\'t select a valid metric"')
        raise Exception('Invalid metric selection. Please select either similarity or distance')

    accuracies = results_obj.accuracy['test']

    # create the figure
    fig = plt.figure(figsize=(10, 5))
    plt.scatter(accuracies, clipped, color='b', label='clipped')
    plt.scatter(accuracies, unclipped, color='r', label='unclipped')

    plt.xlabel(f'Test Accuracy', fontsize=axis_fontsize)
    plt.ylabel(f'{quantity} {metric}', fontsize=axis_fontsize)

    plt.title(f'{experiment_name} {quantity} {metric} vs test accuracy\n(Layer {layer})', fontsize=title_fontsize)

    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')

    plt.legend(loc=legend_loc, fontsize=axis_fontsize)
    plt.tick_params(axis='both', which='both', labelsize=axis_fontsize)
    plt.show()
    return


################################################################### Reshape Results by Layer
def reshape_multilayer_results_dict(original_dict, layer_numbers=None):
    """
    reshapes the results as they are entered in a Perturbation.run_perturbation_training execution

    :param original_dict: dict('quantity': list(list(float))) where each list(float) are the results for multiple layers
    :return: dict(layer: dict(quantity : list(float)) where each list(float) are the results for the layer over all
             perturbation trials
    """
    n_layers = len(original_dict['weights'][0]) # the number of layers is the number of entries for one perturbation lvl

    layer_dicts = []
    for i in range(n_layers):
        layer_results_weights = [original_dict['weights'][k][i] for k in range(len(original_dict['weights']))]
        layer_results_activations = [original_dict['activations'][k][i] for
                                     k in range(len(original_dict['activations']))]
        new_dict = {'activations': layer_results_activations,
                    'weights': layer_results_weights}

        layer_dicts.append(new_dict)

    # set the new keys
    layer_keys = layer_numbers if layer_numbers else list(range(1, n_layers+1))

    # set the reshaped dictionary
    reshaped = dict(zip(layer_keys, layer_dicts))
    return reshaped

def reshape_multilayer_dimensions(original_dimensions_list, layer_numbers=None):
    n_layers = len(original_dimensions_list[0]) # the number of layers is the number of entries for one level
    layer_keys = layer_numbers if layer_numbers else list(range(1, n_layers+1))

    layer_lists = []
    for i in range(n_layers):
        layer_dims = [original_dimensions_list[k][i] for k in range(len(original_dimensions_list))]
        layer_lists.append(layer_dims)

    # set the reshaped dictionary
    reshaped = dict(zip(layer_keys, layer_lists))

    return reshaped

################################################################### Convert quantity output from netsim
# starts as dict('weights' : quantities for each layer, 'activation': quantities for each layer)
# where the quantities are those required for calculating the distance and the similarities
def convert_dist_component_dict(component_dict, quantity):
    # get all the of the weight things
    quant_vals = component_dict[quantity]
    tr1 = [layer[0] for layer in quant_vals] # these are now in order of layer
    tr2 = [layer[1] for layer in quant_vals]
    nnorm = [layer[2] for layer in quant_vals]

    return tr1, tr2, nnorm


###################################
### Multi-Trial Results Handler ###
###################################

def check_if_ticks_are_set_already(results_list):
    """

    :param results_list: list or tuple(PerturbationResults)
    :return: ticks     : list(numerical) or None : checks and returns the ticks from the results lists
    """
    # getting ticks
    obj_ticks = []  # [res.ticks for res in results_list if res.ticks is not None]
    for res in results_list:
        if res.ticks is not None:
            obj_ticks.append(res.ticks)

    preset_ticks = None
    if all(x == obj_ticks[0] for x in obj_ticks):
        preset_ticks = obj_ticks[0]

    return preset_ticks


def reshape_single_attribute_list(results_attribute_list):
    """

    :param results_attribute_list: list(list(numerical) : results for a single attribute for a single layer from all
                                                          results objects in a collection (repeated trials)
    :return: tuple(tuple(numerical))    :   the reshaped results
    """
    reshaped = []
    for k in range(len(results_attribute_list[0])):
        single_trial_list = []
        for res in results_attribute_list:
            single_trial_list.append(res[k])
        reshaped.append(tuple(single_trial_list))

    return reshaped


def reshape_single_attribute_dict(results_attribute_dicts):
    dict_keys = tuple(results_attribute_dicts[0].keys())

    all_keys_results = []
    for k in dict_keys:
        all_trials_results = [res_dict[k] for res_dict in results_attribute_dicts]
        key_results = reshape_single_attribute_list(all_trials_results)
        all_keys_results.append(key_results)

    reshaped = dict(zip(dict_keys, all_keys_results))
    return reshaped


def reshape_dimensions_accuracy_repeated_trials(dimensions_dicts):
    """

    :param dimensions_dicts: dict : keys should be the layers examined for the repeated trial experiments
    :return: dimensions_reshaped: dict  : keys are the same as input; the layers examined. For each layer, for each
                                          trial, the results are given as a tuple of all the results from the trial
    """
    dimensions_reshaped = reshape_single_attribute_dict(dimensions_dicts)

    return dimensions_reshaped


def reshape_repeated_trials_measurement(attribute_dicts):
    """
    takes in dictionary results for one attribute (such as similarity measurements) and returns a dictionary where the
    results contain all the data from all the runs of the experiment for each trial/level of perturbation

    :param attribute_dicts:
    :return:
    """
    layers = tuple(attribute_dicts[0].keys())

    layer_results = []
    for el in layers:
        layer_dict_list = [single_dict[el] for single_dict in attribute_dicts]
        layer_dict = reshape_single_attribute_dict(layer_dict_list)

        layer_results.append(layer_dict)

    reshaped = dict(zip(layers, layer_results))
    return reshaped


def reshape_repeated_trials_attributes(results_list):
    layers = tuple(results_list[0].similarities.keys())

    # reshape the accuracies
    acc_dicts = [res.accuracy for res in results_list]
    accuracies = reshape_dimensions_accuracy_repeated_trials(acc_dicts)

    # reshape the dimensions
    dim_trials = [res.dimensions_trials for res in results_list]
    dimensions_trials = reshape_dimensions_accuracy_repeated_trials(dim_trials)
    ## baseline dimensions
    dims_base = []
    for i in range(len(results_list[0].dimensions_baseline)):
        layer_dims = [res.dimensions_baseline[i] for res in results_list]
        dims_base.append(tuple(layer_dims))
    dimensions_base = dict(zip(layers, dims_base))

    # reshape everything else
    ## reshape similarities
    sim_u_dicts = [res.similarities for res in results_list]
    similarities_unclipped = reshape_repeated_trials_measurement(sim_u_dicts)
    sim_c_dicts = [res.similarities_clipped for res in results_list]
    similarities_clipped = reshape_repeated_trials_measurement(sim_c_dicts)

    ## reshape distances
    dist_u_dicts = [res.distances for res in results_list]
    distances_unclipped = reshape_repeated_trials_measurement(dist_u_dicts)
    dist_c_dicts = [res.distances_clipped for res in results_list]
    distances_clipped = reshape_repeated_trials_measurement(dist_c_dicts)

    ## reshape baseline trace
    tr_base_u_dict = [res.baseline_trace for res in results_list]
    trace_baseline_unclipped = reshape_repeated_trials_measurement(tr_base_u_dict)
    tr_base_c_dict = [res.baseline_trace_clipped for res in results_list]
    trace_baseline_clipped = reshape_repeated_trials_measurement(tr_base_c_dict)

    ## reshape experimental trace
    tr_exp_u_dict = [res.experiment_trace for res in results_list]
    trace_experiment_unclipped = reshape_repeated_trials_measurement(tr_exp_u_dict)
    tr_exp_c_dict = [res.experiment_trace_clipped for res in results_list]
    trace_experiment_clipped = reshape_repeated_trials_measurement(tr_exp_c_dict)

    ## reshape nuclear norms
    nnorm_u_dict = [res.nuclear_norm for res in results_list]
    nuclear_norm_unclipped = reshape_repeated_trials_measurement(nnorm_u_dict)
    nnorm_c_dict = [res.nuclear_norm_clipped for res in results_list]
    nuclear_norm_clipped = reshape_repeated_trials_measurement(nnorm_c_dict)

    reshaped = (accuracies, dimensions_trials, dimensions_base, similarities_unclipped, similarities_clipped,
                distances_unclipped, distances_clipped, trace_baseline_unclipped, trace_baseline_clipped,
                trace_experiment_unclipped, trace_experiment_clipped, nuclear_norm_unclipped, nuclear_norm_clipped)

    return reshaped


# class RepeatedTrialsResults:
#     """
#     Object that handles the results from repeated trials with the same configuration.
#     """

#     def __init__(self, results_list, experiment_name='Experiment', xlabel='Perturbation Level', ticks=None):
#         """

#         :param results_list :   list or tuple(PerturbationResults)   : the results objects to process
#         """
#         self.results = results_list
#         self.layers = tuple(results_list[0].similarities.keys())

#         # setting the ticks
#         ticks = ticks if ticks else check_if_ticks_are_set_already(results_list)
#         if not ticks:
#             print('Default integer ticks set. Please set experiment ticks before plotting for accurate figures.')
#             ticks = tuple(range(len(results_list[0].similarities[self.layers[0]])))

#         # descriptors of the experiment
#         self.name = experiment_name
#         self.xlabel = xlabel
#         self.ticks = ticks
#         self.descriptions = tuple(set([res.description for res in results_list]))
#         ## Measurements taken
#         ### Reshaped with reshaping function
#         (self.accuracy,
#          self.dimensions_exp, self.dimensions_baseline,
#          self.similarities, self.similarities_clipped, self.distances, self.distances_clipped,
#          self.trace_baseline, self.trace_baseline_clipped, self.trace_exp, self.trace_exp_clipped,
#          self.nuclear_norm, self.nuclear_norm_clipped) = (reshape_repeated_trials_attributes(results_list))

#         return

#     def set_name(self, name):
#         self.name = name
#         return f'Successfully set experiment name to {name}'
#     def set_ticks(self, ticks):
#         self.ticks = ticks
#         return 'Successfully set ticks'

#     def plot_effective_dimensions(self, layers=tuple([1]), percent_interval=90, legend_loc='best', xlog=False, ylog=False):
#         plot_effective_dimensions_repeated(self.dimensions_exp, self.dimensions_baseline, ticks=self.ticks, results_list=self.results, layers=layers, percent_interval=percent_interval, experiment_name=self.name, xlabel=self.xlabel, legend_loc=legend_loc, xlog=xlog, ylog=ylog)
#         return

#     def plot_trajectories(self, metric='similarity', layers=tuple([1]), quantity='weights', percent_interval=90, legend_loc='best', xlog=False, ylog=True, ylim=None):
#         for layer in layers:
#             plot_trajectories_repeated(self, metric=metric, layer=layer, quantity=quantity, percent_interval=percent_interval, experiment_name=self.name, xlabel=self.xlabel, legend_loc=legend_loc, xlog=xlog, ylog=ylog, ylim=ylim)
#         return

#     def plot_metric_component_quantities(self, quantity='weights', layers=tuple([1]), clipped=True, legend_loc='best', yrange=None, xlog=False, ylog=True, percent_interval=90):
#         for layer in layers:
#             plot_component_quantity_trajectory_repeated(self, percent_interval=percent_interval, ticks=self.ticks, quantity=quantity, layer=layer, clipped=clipped, legend_loc=legend_loc, yrange=yrange, xlog=xlog, ylog=ylog, xlabel=self.xlabel, experiment_name=self.name)
#         return



# def avg_and_range_single(trial_list, lo_val=0.05, hi_val=0.95, percent_interval=None):
#     if percent_interval:
#         lo_val, hi_val = 50 - (percent_interval / 2), 50 + (percent_interval / 2)

#     qlo = np.percentile(trial_list, lo_val)
#     qhi = np.percentile(trial_list, hi_val)
#     avg = np.mean(trial_list)

#     return avg, (qlo, qhi)


# def avg_and_errors(plot_list, lo_val=0.05, hi_val=0.95, percent_interval=None):
#     """

#     :param plot_list:   list(tuple(numerical))  : the list of repeated observations at each level of an experiment
#     :param lo_val:      float                   : the quantile of the lower error bound
#     :param hi_val:      float                   : the quantile of the higher error bound
#     :return: avg:       tuple(float)            : the average of the observations at each level
#     :return: (qlo, qhi):(tuple(float), tuple(float) : the lower and higher error of observations at each level, ready to
#                                                       be entered into matplotlib.pyplot.errorbar
#     """
#     if percent_interval:
#         lo_val, hi_val = 50 - (percent_interval / 2), 50 + (percent_interval / 2)

#     qlo = [np.percentile(level, lo_val) for level in plot_list]
#     qhi = [np.percentile(level, hi_val) for level in plot_list]
#     avg = tuple([np.mean(level) for level in plot_list])

#     qlo = tuple([avg[i] - qlo[i] for i in range(len(avg))])
#     #qhi = tuple([qhi[i] - avg[i] for i in range(len(avg))])
#     qhi = tuple([qhi[i] - avg[i] for i in range(len(avg))])

#     return avg, (qlo, qhi)


# def plot_effective_dimensions_repeated(dimensions_exp, dimensions_baseline, ticks=None, results_list=None,
#                                        layers=tuple([1]), percent_interval=90,
#                                        experiment_name='', xlabel=None, legend_loc='best', xlog=False, ylog=False):
#     ticks = ticks if ticks else tuple(range(len(dimensions_exp)))
#     xlabel = xlabel if xlabel else 'Perturbation Amount'
#     q_lo, q_hi = 50-(percent_interval/2),  50+(percent_interval/2)

#     for layer in layers: # for each layer,
#         # get the averages to plot and the errors
#         y_plot, errors = avg_and_errors(dimensions_exp[layer], lo_val=q_lo, hi_val=q_hi)
#         #print(errors)

#         y_base, base_range = avg_and_range_single(dimensions_baseline[layer], lo_val=q_lo, hi_val=q_hi)
#         print(base_range[0], base_range[1])

#         fig = plt.figure(figsize=(10, 5)) # create the figure

#         # plot the averages with the error bars
#         plt.errorbar(ticks, y_plot, yerr=errors, fmt='m-o', capsize=3, ecolor=colors.to_rgba('m', 0.5),
#                      label=f'effective dimensions ({percent_interval}% interval)')

#         # plot the baseline region
#         plt.hlines(y_base, min(ticks), max(ticks), color='green', linestyles=':', linewidth=1)
#         plt.fill_between(x=(float(min(ticks)),float(max(ticks))), y1=base_range[0], y2=base_range[1], color='green', alpha=0.25, label=f'baseline dimensions ({percent_interval}% interval)')

#         # plot the individual results if included
#         if results_list:
#             for res in results_list:
#                 plt.plot(ticks, res.dimensions_trials[layer], color=colors.to_rgba('m', 0.1))

#         plt.title(f'{experiment_name} Effective Dimensionality\n(Layer {layer})', fontsize=title_fontsize)
#         plt.ylabel('Effective Dimensions', fontsize=axis_fontsize)
#         plt.xlabel(f'{xlabel}', fontsize=axis_fontsize)

#         if xlog:
#             plt.xscale('log')
#         if ylog:
#             plt.yscale('log')

#         plt.legend(loc=legend_loc, fontsize=axis_fontsize)
#         plt.show()
#     return


# def plot_trajectories_repeated(repeated_results_object, metric='similarity', layer=1, quantity='weights', percent_interval=0.9, experiment_name='', xlabel='Perturbation Level', legend_loc='best', xlog=False, ylog=True, ylim=None):
#     # set the ticks
#     ticks = repeated_results_object.ticks

#     # get all the metric measurements
#     if metric == 'similarity':
#         unclipped = getattr(repeated_results_object, 'similarities')
#         clipped = getattr(repeated_results_object, 'similarities_clipped')

#         unclipped_individual = tuple([res.similarities[layer][quantity] for res in repeated_results_object.results])
#         clipped_individual = tuple([res.similarities_clipped[layer][quantity] for res in repeated_results_object.results])

#         if quantity == 'weights':
#             id_baseline = way_id_sim
#             unclipped_baseline_key = 'wsu'
#         else:
#             id_baseline = act_id_sim
#             unclipped_baseline_key = 'as'

#     elif metric == 'distance':
#         unclipped = getattr(repeated_results_object, 'distances')
#         clipped = getattr(repeated_results_object, 'distances_clipped')
#         unclipped_individual = tuple([res.distances[layer][quantity] for res in repeated_results_object.results])
#         clipped_individual = tuple(
#             [res.distances_clipped[layer][quantity] for res in repeated_results_object.results])

#         if quantity == 'weights':
#             id_baseline = way_id_dist
#             unclipped_baseline_key = 'wdu'
#         else:
#             id_baseline = act_id_dist
#             unclipped_baseline_key = 'ad'
#     else:
#         os.system('say "You absolute bufoon, you didn\'t select a valid metric"')
#         raise Exception('Invalid metric selection. Please select either similarity or distance')

#     # get the correct layer and quantity
#     unclipped = unclipped[layer][quantity]
#     clipped = clipped[layer][quantity]

#     # get the error bar range
#     # get the averages and error bars
#     y_unclipped, errors_unclipped = avg_and_errors(unclipped, percent_interval=percent_interval)
#     #print(errors_unclipped)
#     y_clipped, errors_clipped = avg_and_errors(clipped, percent_interval=percent_interval)
#     #print(errors_clipped)

#     # Making the Figure
#     fig = plt.figure(figsize=(10,5))

#     # Plotting the aggregate trajectories
#     plt.errorbar(ticks, y_unclipped, yerr=errors_unclipped, fmt='ro', ecolor=colors.to_rgba('red', 0.5), label=f'unclipped ({percent_interval}% interval)', markersize=5, capsize=3)
#     plt.errorbar(ticks, y_clipped, yerr=errors_clipped, fmt='bo', ecolor=colors.to_rgba('blue', 0.5), label=f'clipped ({percent_interval}% interval)', markersize=5, capsize=3)

#     # Plotting individual trajectories
#     for k in range(len(repeated_results_object.results)):
#         plt.plot(ticks, clipped_individual[k], color=colors.to_rgba('blue', 0.075))
#         plt.plot(ticks, unclipped_individual[k], color=colors.to_rgba('red', 0.075))

#     # baselines
#     ## identical weights baseline
#     plt.hlines(id_baseline, min(ticks), max(ticks), colors='orange', linestyles=':', label='identical weights')
#     plt.hlines(np.mean(quantities[unclipped_baseline_key][2]), min(ticks), max(ticks), colors='red', linestyles=':', label='unclipped random init')

#     plt.title(f'{experiment_name} {quantity} {metric} trajectories\nLayer {layer}', fontsize=title_fontsize)
#     plt.xlabel(xlabel, fontsize=axis_fontsize)
#     plt.ylabel(f'{metric}', fontsize=axis_fontsize)

#     if ylim:
#         plt.ylim(ylim)

#     if xlog:
#         plt.xscale('log')
#     if ylog:
#         plt.yscale('log')

#     plt.legend(loc=legend_loc, fontsize=axis_fontsize)
#     plt.show()
#     return


# def plot_component_quantity_trajectory_repeated(repeated_results_object : RepeatedTrialsResults, percent_interval=90, ticks=None, quantity='weights', layer=1, clipped=True, experiment_name='', xlabel='Perturbation Level', legend_loc='best', yrange=None, xlog=False, ylog=True):
#     yrange = yrange if yrange else default_ranges[quantity][0]
#     ticks = ticks if ticks else repeated_results_object.ticks

#     if len(yrange) != 2:
#         os.system('say "Hey dingus, I cant set an axis limit with only one value! Idiot"')
#         raise Exception('Please ensure there are exactly two values for each of the axis limit settings. This can be'
#                         'as either a list or as a tuple.')

#     # Get each of the quantities
#     distance = repeated_results_object.distances_clipped[layer][quantity] if clipped else repeated_results_object.distances[layer][quantity]
#     exp_trace = repeated_results_object.trace_exp_clipped[layer][quantity] if clipped else repeated_results_object.trace_exp[layer][quantity]
#     base_trace = repeated_results_object.trace_baseline_clipped[layer][quantity][0] if clipped else repeated_results_object.trace_baseline[layer][quantity][
#         0]
#     nuc_norm = repeated_results_object.nuclear_norm_clipped[layer][quantity] if clipped else repeated_results_object.nuclear_norm[layer][quantity]

#     # calculate the ranges for each of the bois
#     y_dist, err_dist = avg_and_errors(distance, percent_interval=percent_interval)
#     y_exp_trace, err_exp_trace = avg_and_errors(exp_trace, percent_interval=percent_interval)
#     y_norm, err_norm = avg_and_errors(nuc_norm, percent_interval=percent_interval)
#     # baseline region
#     base_trace_line, base_trace_region = avg_and_range_single(base_trace, percent_interval=percent_interval)

#     # set the color of the plotting
#     meas_c = 'b' if clipped else 'r'

#     # create the figure
#     fig = plt.figure(figsize=(10, 5)) #({percent_interval}% interval)

#     # plot the error bars quantities
#     ## distance
#     plt.errorbar(ticks, y_dist, yerr=err_dist, fmt=f'{meas_c}-o', ecolor=colors.to_rgba(meas_c, 0.5), capsize=3, label=f'distance ({percent_interval}% interval)')

#     ## trace exp
#     plt.errorbar(ticks, y_exp_trace, yerr=err_exp_trace, fmt=f'm:^', ecolor=colors.to_rgba('m', 0.5), capsize=3, label=(r'$Tr(C_{exp})$'+f'({percent_interval}% interval)'))

#     ## base trace h lines fill between region
#     plt.hlines(base_trace_line, min(ticks), max(ticks), colors='orange', linestyles=':', label=(r'$Tr(C_{ref})$'+f'({percent_interval}% interval)'))
#     plt.fill_between((float(min(ticks)), float(max(ticks))), base_trace_region[0], base_trace_region[1], color='orange', alpha=0.25)

#     ## nuc norm
#     plt.errorbar(ticks, y_norm, yerr=err_norm, fmt='g:^', ecolor=colors.to_rgba('g', 0.5), capsize=3, label=(r'$||C_{ref}^{1/2} C_{exp}^{1/2}||_{nuc}$'+f'({percent_interval}% interval)'))

#     # plot the individual trajectories
#     for res in repeated_results_object.results:
#         if clipped:
#             plt.plot(ticks, res.distances_clipped[layer][quantity], color='b', alpha=0.075)
#             plt.plot(ticks, res.experiment_trace_clipped[layer][quantity], color='m', alpha=0.075)
#             plt.plot(ticks, res.nuclear_norm_clipped[layer][quantity], color='green', alpha=0.075)
#         else:
#             plt.plot(ticks, res.distances[layer][quantity], color='b', alpha=0.075)
#             plt.plot(ticks, res.experiment_trace[layer][quantity], color='m', alpha=0.075)
#             plt.plot(ticks, res.nuclear_norm[layer][quantity], color='green', alpha=0.075)

#     # labelling etc
#     clip_title = 'clipped' if clipped else ''
#     plt.title(f'{experiment_name} {clip_title} {quantity} metric component trajectories', fontsize=title_fontsize)
#     plt.ylabel(f'Total Variation/Distance', fontsize=axis_fontsize)
#     plt.xlabel(xlabel, fontsize=axis_fontsize)

#     if xlog:
#         plt.xscale('log')
#     if ylog:
#         plt.yscale('log')

#     plt.ylim(yrange)

#     plt.legend(fontsize=axis_fontsize)
#     plt.show()

#     return


# def plot_metric_vs_accuracy_repeated(repeated_results_object, metric='similarity', layer=1, quantity='weights', experiment_name='', legend_loc='best', xlog=False, ylog=False):
#     # check metric
#     if metric == 'similarity':
#         clipped_key = 'similarities_clipped'
#         unclipped_key = 'similarities'
#     elif metric == 'distance':
#         clipped_key = 'distances_clipped'
#         unclipped_key = 'distances'
#     else:
#         os.system('say "You absolute bufoon, you didn\'t select a valid metric"')
#         raise Exception('Invalid metric selection. Please select either similarity or distance')

#     fig = plt.figure(figsize=(10, 5))
#     for res in repeated_results_object.results:
#         clipped = getattr(res, clipped_key)[layer][quantity]
#         unclipped = getattr(res, unclipped_key)[layer][quantity]
#         accuracies = getattr(res, 'accuracy')['test']

#         plt.scatter(accuracies, clipped, color='b')
#         plt.scatter(accuracies, unclipped, color='r')

#     plt.show()

#     return

