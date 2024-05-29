import spectral_analysis as spec
import alignment as align
import network_similarity as sim
import class_splitter as cs
import neural_network as nn_mod

import pickle
from tqdm import tqdm
from tqdm.notebook import tqdm
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
import torch.utils.data.dataset
from torchvision import datasets
from torchvision import transforms

####################################
### Baseline Pickle Measurements ###
####################################
# load in the baseline identical measurements
#
with open('/Users/mnzk/Documents/40-49. Research/42. Nets/42.97. Library/pickle_vars/baseline_identical.pkl', 'rb') as f:
    id_baseline = pickle.load(f)
way_id_sim = id_baseline['way_sim']
act_id_sim = id_baseline['act_sim']
way_id_dist = id_baseline['way_dist']
act_id_dist = id_baseline['act_dist']

# load in the quantity baselines
with open('/Users/mnzk/Documents/40-49. Research/42. Nets/42.97. Library/pickle_vars/baseline_quantiles.pkl', 'rb') as f:
    quantities = pickle.load(f)

###########################
### Perturbation Object ###
###########################
class PerturbationConfig:
    """
    Holds the configuration for a perturbation experiment
    """

    def __init__(self,
                 n_classes=10,
                 batch_size=64,
                 dataset_class=datasets.CIFAR10,
                 columns=None, rows=None,
                 val=225,
                 intensity=False,
                 noise=False, var=0,
                 random_pixels=None,
                 name=None
                 ):
        self.model_names = ('unperturbed', name) if name else ('unperturbed', 'perturbed')
        # get the datasets
        # unperturbed dataset
        self.loaders_normal = subset_class_loader(list(range(n_classes)),
                                                  batch_size=batch_size)
        self.dataset_class = dataset_class
        self.batch_size = batch_size

        self.n_classes = n_classes

        # set the perturbation parameters
        self.columns = columns
        self.rows = rows
        self.random_pixels = random_pixels

        # get the number of trials from these
        lengths = [len(i) if i is not None else 0 for i in (columns, rows, random_pixels)]
        self.n_trials = max(lengths)
        # test to see if this is correct
        if not all((x == max(lengths) or x == 0) for x in lengths):
            raise Exception('Number of perturbations does not match between modes.'
                            'Please ensure each list of values is of the same length')

        # other settings
        self.val = val
        self.intensity = intensity
        self.noise = noise
        self.var = var
        return

class PerturbationResults:
    """
    holds and plots perturbation experiment results
    """
    def __init__(self,
                 similarities,
                 similarities_clipped,
                 distances,
                 distances_clipped):
        self.similarities = similarities
        self.similarities_clipped = similarities_clipped

        self.distances = distances
        self.distances_clipped = distances_clipped
        return

    def plot_trajectories(self,
                          wsim_order=(2, 4, 3, 1, 0),
                          asim_order=(2, 3, 0, 1),
                          wdist_order=(3, 0, 1, 4, 2),
                          adist_order=(0, 1, 3, 2),
                          ws_loc='upper left',
                          as_loc='upper left',
                          wd_loc='lower right',
                          ad_loc='upper left'):
        plot_result_trajectories(self.similarities, self.similarities_clipped, self.distances, self.distances_clipped,
                                 wsim_order=wsim_order, asim_order=asim_order,
                                 wdist_order=wdist_order, adist_order=adist_order,
                                 ws_loc=ws_loc, as_loc=as_loc, wd_loc=wd_loc, ad_loc=ad_loc)
        return


def run_perturbation_iteration(baseline_model : spec.spectrum_analysis,
                               layers,
                               n_classes=10,
                               batch_size=64,
                               dataset_class=datasets.CIFAR10,
                               columns=None, rows=None,
                               val=225,
                               intensity=False,
                               noise=False, var=0,
                               random_pixels=None,
                               name=None,

                               j=None,
                               w_clip_val=30, a_clip_val=64, plot=True
                               ):
    start = time.time()
    model_names = ('unperturbed', name) if name else ('unperturbed', 'perturbed')
    n_layers = len(baseline_model.n_neurons)

    lens = [0 if k is None else k for k in (columns, rows, random_pixels)]
    j = j if j else max(lens)

    # create the dataloaders
    trial_loaders = subset_class_loader([], mod_ind=list(range(n_classes)),
                                        batch_size=batch_size, dataset_class=dataset_class,
                                        columns=columns, rows=rows,
                                        val=val, intensity=intensity,
                                        noise=noise, var=var, random_pixels=random_pixels)
    # show the dataset
    dataiter = iter(trial_loaders[0])
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[0:5]))

    # train the model
    trial_model = spec.spectrum_analysis(baseline_model.n_neurons)  # create the model
    trial_model.train(trial_loaders[0], trial_loaders[1],
                      baseline_model.n_epochs, grain=50000, ep_grain=baseline_model.n_epochs)
    trial_model.get_effective_dimensions()

    # saving accuracy and EDs
    accuracy_trials = trial_model.val_history[-1]
    dimensions_trials = [trial_model.effective_dimensions[j][-1] for j in range(n_layers)]

    # create the similarity object
    simobj = sim.network_comparison(baseline_model, trial_model, names=model_names)

    # do the similarity analysis
    # get the alignments
    simobj.compute_alignments(baseline_model.train_loader, layers)
    simobj.compute_cossim()

    # get the metrics
    ###### DISTANCES
    ## clipped values
    act, way = simobj.network_distance(w_clip=w_clip_val, a_clip=a_clip_val)
    act_dist = act[0]
    way_dist = way[0]
    ## unlcipped values
    act, way = simobj.network_distance(w_clip=None)
    act_dist_clip = act[0]
    way_dist_clip = way[0]

    ###### SIMILARITIES
    ## clipped values
    act, way = simobj.network_distance(w_clip=w_clip_val,
                                       a_clip=a_clip_val,
                                       sim=True)
    act_sim = act[0]
    way_sim = way[0]
    ## unclipped values
    act, way = simobj.network_distance(w_clip=None, sim=True)
    act_sim_clip = act[0]
    way_sim_clip = way[0]

    if plot:
        simobj.plot_sims(quantities=['activations'], clips=[int(a_clip_val * 1.5)] * n_layers,
                         filename_append=f'p--{j}',
                         plot_clip=a_clip_val)
        simobj.plot_sims(quantities=['weights'], clips=[w_clip_val + 20] * n_layers,
                         filename_append=f'p--{j}',
                         plot_clip=w_clip_val)

    print(f'TOTAL TIME: {time.time() - start} seconds')

    return (act_dist, way_dist, act_dist_clip, way_dist_clip,
            act_sim, way_sim, act_sim_clip, way_sim_clip,
            accuracy_trials, dimensions_trials)


def add_perturbation_results_to_dict(perturbation_iteration_results,
                                     sim_uncl_dict,
                                     sim_clip_dict,
                                     dist_uncl_dict,
                                     dist_clip_dict,
                                     accuracy_list,
                                     dimensions_list):

    dist_uncl_dict['activations'].append(perturbation_iteration_results[0])
    dist_uncl_dict['weights'].append(perturbation_iteration_results[1])

    dist_clip_dict['activations'].append(perturbation_iteration_results[2])
    dist_clip_dict['weights'].append(perturbation_iteration_results[3])

    sim_uncl_dict['activations'].append(perturbation_iteration_results[4])
    sim_uncl_dict['weights'].append(perturbation_iteration_results[5])

    sim_clip_dict['activations'].append(perturbation_iteration_results[6])
    sim_clip_dict['weights'].append(perturbation_iteration_results[7])

    accuracy_list.append(perturbation_iteration_results[8])
    dimensions_list.append(perturbation_iteration_results[9])

    return sim_uncl_dict, sim_clip_dict, dist_uncl_dict, dist_clip_dict, accuracy_list, dimensions_list


class Perturbation:

    def __init__(self,
                 n_classes=10,
                 batch_size=64,
                 dataset_class=datasets.CIFAR10,
                 columns=None, rows=None,
                 val=255,
                 intensity=False,
                 noise=False, var=None,
                 random_pixels=None,
                 name=None):
        """
        Does not currently handle intensity modifications!

        :param n_classes:
        :param batch_size:
        :param dataset_class:
        :param columns: list of numbers of columns to perturb
        :param rows: list of numbers of rows to perturb
        :param val: the (mean) value to replace the pixel(s) with
        :param intensity: whether or not to clip based on intensity
        :param noise: whether or not to vary the value according to a gaussian distribution
        :param var: if noise, the variance of the distribution
        :param random_pixels: list of the number of random pixels to perturb
        """
        self.model_names = ('unperturbed', name) if name else ('unperturbed', 'perturbed')
        # get the datasets
        # unperturbed dataset
        self.loaders_normal = subset_class_loader(list(range(n_classes)),
                                                  batch_size=batch_size)
        self.dataset_class = dataset_class
        self.batch_size = batch_size

        self.n_classes = n_classes

        # set the perturbation parameters
        self.columns = columns
        self.rows = rows
        self.random_pixels = random_pixels

        # get the number of trials from these
        lengths = [len(i) if i is not None else 0 for i in (columns, rows, random_pixels, var)]
        self.n_trials = max(lengths)
        # test to see if this is correct
        if not all((x == max(lengths) or x == 0) for x in lengths):
            raise Exception('Number of perturbations does not match between modes.'
                            'Please ensure each list of values is of the same length')

        # other settings
        self.val = val
        self.intensity = intensity
        self.noise = noise
        self.var = var

        # results tracking
        self.accuracy_baseline = None
        self.accuracy_trials = []

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

        # running parameters (set later)
        self.baseline_model = None
        self.n_neurons = None
        self.n_epochs = None

        self.layers = None

        return

    def set_baseline(self, n_epochs, n_neurons):
        # create the model
        baseline_model = spec.spectrum_analysis(n_neurons)
        self.n_neurons = n_neurons

        # train the model for n_epochs
        self.n_epochs = n_epochs # save so we can use the same number for later

        baseline_train = self.loaders_normal[0]
        baseline_val = self.loaders_normal[1]

        # training
        baseline_model.train(baseline_train, baseline_val, n_epochs, grain=50000, ep_grain=n_epochs)

        # adding the accuracy to the record
        self.accuracy_baseline = baseline_model.val_history[-1]
        # get the weight covariances
        baseline_model.get_weight_covs()
        # and effective dimensions
        baseline_model.get_effective_dimensions()
        self.dimensions_baseline = [baseline_model.effective_dimensions[j][-1] for j in range(len(n_neurons))]

        self.baseline_model = baseline_model
        return

    def run_perturbation_training(self, layers, perturbation_level, w_clip_val=30, a_clip_val=64, plot=True):
        start = time.time()
        # for each trial
        j = perturbation_level
        #for j in tqdm(range(trial_start, trial_end), desc='running the perturbations'):
        # first, set up the indices if there are rows and or columns
        j_col = self.columns[j] if self.columns else None
        j_row = self.rows[j] if self.rows else None
        j_pix = self.random_pixels[j] if self.random_pixels else None
        j_var = self.var[j] if self.var else None

        # create the dataloader
        trial_loaders = subset_class_loader([], mod_ind=list(range(self.n_classes)),
                                            columns=j_col, rows=j_row,
                                            val=self.val, intensity=self.intensity,
                                            noise=self.noise, var=j_var, random_pixels=j_pix)

        # show the dataset
        display_dataloader_images(trial_loaders[0], 5)

        # train the model
        trial_model = spec.spectrum_analysis(self.n_neurons) # create the model
        trial_model.train(trial_loaders[0], trial_loaders[1],
                            self.n_epochs, grain=50000, ep_grain=self.n_epochs) # train the model
        trial_model.get_effective_dimensions() # get the effective dimensions
        self.accuracy_trials.append(trial_model.val_history[-1]) # add the accuracy results
        self.dimensions_trials.append([trial_model.effective_dimensions[j][-1] for j in range(len(self.n_neurons))])
                    # add the effective dimensions

        # create the similarity object
        simobj = sim.network_comparison(self.baseline_model, trial_model, names=self.model_names)

        # do the similarity analysis
        # get the alignments
        simobj.compute_alignments(self.loaders_normal[0], layers) # compute alignments for each layer
        simobj.compute_cossim()

        # get the metrics and add them to the correct list
        ###### DISTANCES
        ## clipped values
        act, way, quants = simobj.network_distance(w_clip=w_clip_val, a_clip=a_clip_val, return_quantities=True) # these come back with all layers
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
        #del trial_model
        #del simobj
        print(f'TOTAL TIME: {(time.time() - start):.2f} seconds')
        return

    def reshape_results(self):
        #if len(self.layers) == 1:
        #    raise Exception(f'This perturbation is a single layer analysis. Cannot perform multilayer results reshape')
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
        print('Reshaped results for multilayer format')

        return

    def plot_trajectories(self, layers=None,
                          wsim_order=(2, 4, 3, 1, 0),
                          asim_order=(2, 3, 0, 1),
                          wdist_order=(3, 0, 1, 4, 2),
                          adist_order=(0, 1, 3, 2),
                          ws_loc='upper left',
                          as_loc='upper left',
                          wd_loc='lower right',
                          ad_loc='upper left',
                          xticks =None,
                          xlabadd =None,
                          ylabadd =None,
                          titleadd ='',
                          ylog=True):
        layers = layers if layers else self.layers

        for layer in layers:
            print(f'Plotting results for layer {layer}')
            plot_result_trajectories(self.similarities[layer],
                                     self.similarities_clipped[layer],
                                     self.distances[layer],
                                     self.distances_clipped[layer],
                                     wsim_order=wsim_order, asim_order=asim_order,
                                     wdist_order=wdist_order, adist_order=adist_order,
                                     ws_loc=ws_loc, as_loc=as_loc, wd_loc=wd_loc, ad_loc=ad_loc,
                                     xticks=xticks, xlabadd=xlabadd, ylabadd=ylabadd,
                                     titleadd = f'{titleadd} - Layer {layer}',
                                     ylog=ylog)
        return

    def plot_accuracy(self, xticks=None, legend_loc = 'lower left', titleadd=''):
        plot_accuracy_trajectory(self.accuracy_trials, self.accuracy_baseline, xticks=xticks,
                                 legend_loc=legend_loc, titleadd=titleadd)

        return

    def save_trajectories(self, pickle_name, description=''):
        names = ['sims_u', 'sims_c', 'dist_u', 'dist_c', 'acc', 'effdims', 'base_dims',
                 'base_tr_u', 'base_tr_c', 'exp_tr_u', 'exp_tr_c', 'nnorm_u', 'nnorm_c']
        dicts = [self.similarities, self.similarities_clipped,
                 self.distances, self.distances_clipped,
                 self.accuracy_trials, self.dimensions_trials, self.dimensions_baseline,
                 self.baseline_trace, self.baseline_trace_clipped,
                 self.experiment_trace, self.experiment_trace_clipped,
                 self.nuclear_norm, self.nuclear_norm_clipped]
        results = {}

        # create the results dictionary
        for i in range(len(names)):
            results[names[i]] = dicts[i]

        results['README'] = description

        PATH = f'../pickle_vars/{pickle_name}.pkl'

        # save the results
        with open(PATH, 'wb') as f:
            pickle.dump(results, f)


##############################
###### Helper Functions ######
##############################


########################################################### Displaying images
# show the images
# for showing the image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.xticks([])
    plt.yticks([])
    plt.show()


def display_dataloader_images(dataloader, n_images):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[0:n_images]))
    return


############################################################# Generating Datasets
# for generating the datasets
def subset_class_loader(class_indices, batch_size=64,
                        dataset_class=datasets.CIFAR10,
                        mod_ind=None,
                        columns=None, rows=None,
                        val=255,
                        intensity=False,
                        noise=False, var=0,
                        random_pixels=None):
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

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mod_transform = transforms.Compose([#transforms.Normalize(mean=mean, std=std),
                                            dataset_perturbations(column_indices=columns,
                                                                  row_indices=rows,
                                                                  val=val,
                                                                  intensity=intensity,
                                                                  noise=noise, var=var,
                                                                  random_pixels=random_pixels)#,
                                            #transforms.Normalize(mean=mean, std=std)
        ])

        modded_train = MyDataset(m_train_sub, transform=mod_transform)
        modded_val = MyDataset(m_val_sub, transform=mod_transform)

        trainset_sub = torch.utils.data.ConcatDataset([trainset_sub, modded_train])
        valset_sub = torch.utils.data.ConcatDataset([valset_sub, modded_val])

    # get the dataloader
    train_loader_sub = nn_mod.get_dataloader(batch_size, trainset_sub, shuffle=True)
    val_loader_sub = nn_mod.get_dataloader(batch_size, valset_sub, shuffle=False)

    return train_loader_sub, val_loader_sub


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


# dataset purturbations class for the transformations
class dataset_perturbations(object):
    def __init__(self, val=255,
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
        self.val = val
        self.columns = column_indices
        self.rows = row_indices
        self.intensity = intensity

        self.random_pix = random_pixels

        if noise:
            if var == 0:
                raise Exception('If using random masking, please include nonzero variance')
            if self.columns or self.rows:
                self.random = scipy.stats.norm(0, np.sqrt(var))
                self.std = np.sqrt(var)
            else:
                self.random = True
                self.std = np.sqrt(var) if var else 0
        else:
            self.random = None
            self.std = None


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

            if self.random:
                for index in self.columns:
                    for i in range(len(img_tensor[0])):
                        img_tensor[:, i, index] += self.random.rvs(n=1)[0]

        if self.rows:
            for index in self.rows:
                img_tensor[:, index, :] = torch.tensor(self.val)

            if self.random:
                for index in self.rows:
                    for i in range(len(img_tensor[0][0])):
                        img_tensor[:, i, index] += self.random.rvs(size=3)

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

        if self.random and not (self.rows or self.columns):
            img_tensor += torch.randn(img_tensor.size()) * self.std

        if self.intensity:
            img_tensor = torch.clamp(img_tensor, min=0, max=self.val)

        return img_tensor

    def __repr__(self):
        return "Turning Columns and or Rows a Color"


def get_random_pixel_indices(img_shape, total_pix):
    indices_long = np.random.choice(np.arange(img_shape[0]*img_shape[1]), size=total_pix)
    indices = [divmod(indices_long[j], img_shape[1]) for j in range(total_pix)]

    return indices


################################################################# Plotting Results
# Plotting Functions
def plot_result_trajectories(similarities,
                             similarities_clipped,
                             distances,
                             distances_clipped,
                             wsim_order=(2, 4, 3, 1, 0),
                          asim_order=(2, 3, 0, 1),
                          wdist_order=(3, 0, 1, 4, 2),
                          adist_order=(0, 1, 3, 2),
                          ws_loc='upper left',
                          as_loc='upper left',
                          wd_loc='lower right',
                          ad_loc='upper left',
                             xticks=None,
                             xlabadd=None, ylabadd=None,
                             titleadd=None,
                             ylog = True):

    ylabadd = ylabadd if ylabadd else ''
    xlabadd = xlabadd if xlabadd else ''

    x_title = f'Perturbation Amount {xlabadd}'
    #y_title = f'Perturbation Amount {ylabadd}'

    # create the figure for the
    #########################
    ## WEIGHT SIMILARITIES ##
    #########################
    fig = plt.figure(figsize=(10, 5))

    # Get and plot the X sets
    # unclipped similarities
    xu = np.array(similarities['weights'])
    mask = np.isfinite(xu)
    plt.scatter(np.array(range(len(xu)))[mask], xu[mask],
                marker='o',
                color='r',
                label=f'unclipped sim')  #### LEGEND: 0
    # clipped similarities
    xc = np.array(similarities_clipped['weights'])
    mask = np.isfinite(xc)
    plt.scatter(np.array(range(len(xu)))[mask], xc[mask],
                marker='o',
                color='b',
                label=f'clipped sim')  #### LEGEND: 1
    # plot the baselines
    # identical weights distance
    plt.hlines(way_id_sim, 0, len(xu) - 1, colors='orange', linestyles=':', label='identical weights')  #### L2
    # random init, same arch
    plt.hlines(np.mean(quantities['wsu'][2]), 0, len(xu) - 1, colors='r', linestyles=':',
               label='unclipped random init')  #### L3
    plt.hlines(np.mean(quantities['wsc'][2]), 0, len(xu) - 1, colors='b', linestyles=':',
               label='clipped random init')  #### L4

    plt.title(f'Weight cosine similarity trajectories {titleadd}')
    plt.xlabel(x_title)
    plt.ylabel(f'Cosine Similarity {ylabadd}')

    # custom legend order
    handles, labels = plt.gca().get_legend_handles_labels()
    ############## CHANGE ORDER HERE
    order = wsim_order
    if ylog:
        plt.yscale('log')
    #################################### Setting the Ticks
    tick_places = list(range(len(xu)))
    ticknames = xticks if xticks else list(range(len(xu)))
    plt.xticks(tick_places, ticknames)
    plt.legend([handles[i] for i in order],
               [labels[i] for i in order],
               loc=ws_loc)
    plt.show()

    # create the figure for the
    #################
    ## ACTIVATIONS ##
    #################
    fig = plt.figure(figsize=(10, 5))
    # Get and plot the X sets
    xu = np.array(similarities['activations'])
    mask = np.isfinite(xc)
    plt.scatter(np.array(range(len(xu)))[mask], xu[mask],
                marker='o',
                color='r',
                label=f'unclipped sim')  ### Legend 0
    # clipped
    # clipped similarities
    xc = np.array(similarities_clipped['activations'])
    mask = np.isfinite(xc)
    plt.scatter(np.array(range(len(xu)))[mask], xc[mask],
                marker='o',
                color='b',
                label=f'clipped sim')  #### LEGEND: 1
    # plot the baselines
    # indentical weights
    plt.hlines(act_id_sim, 0, len(xu) - 1, colors='orange', linestyles=':', label='identical weights')  #### L2
    # random init, same arch
    plt.hlines(np.mean(quantities['as'][2]), 0, len(xu) - 1, colors='r', linestyles=':', label='random init')  #### L3
    plt.title(f'Activation cosine similarity trajectories {titleadd}')
    plt.xlabel(x_title)
    plt.ylabel(f'Cosine Similarity {ylabadd}')
    # custom legend order
    handles, labels = plt.gca().get_legend_handles_labels()
    ############## CHANGE ORDER HERE
    order = asim_order
    if ylog:
        plt.yscale('log')
    plt.xticks(tick_places, ticknames)
    plt.legend([handles[i] for i in order],
               [labels[i] for i in order],
               loc=as_loc)
    plt.show()

    # plotting the distances
    # plot each of the sets of distances for weights and activations
    # create the figure
    fig = plt.figure(figsize=(10, 5))

    # Get and plot the X sets
    xu = np.array(distances['weights'])
    mask = np.isfinite(xc)
    plt.scatter(np.array(range(len(xu)))[mask], xu[mask],
                marker='o',
                color='r',
                label=f'unclipped dist')  #### Legend 0

    # clipped distances
    xc = np.array(distances_clipped['weights'])
    mask = np.isfinite(xc)
    plt.scatter(np.array(range(len(xu)))[mask], xc[mask],
                marker='o',
                color='b',
                label=f'clipped dist')  #### LEGEND: 1

    # plot the baselines
    # identical weights distance
    plt.hlines(way_id_dist, 0, len(xu) - 1, colors='orange', linestyles=':', label='identical weights')  #### L2

    # random init, same arch
    # plt.hlines(way_rand_dist, 0, lvl, colors='r', linestyles=':', label='unclipped random init') #### L3
    plt.hlines(np.mean(quantities['wdu'][2]), 0, len(xu) - 1, colors='r', linestyles=':',
               label='unclipped random init')  #### L3

    # random init, clipped
    plt.hlines(np.mean(quantities['wdc'][2]), 0, len(xu) - 1, colors='b', linestyles=':',
               label='clipped random init')  #### L4

    plt.title(f'Weights Distance trajectories {titleadd}')
    plt.xlabel(x_title)
    plt.ylabel(f'Distance {ylabadd}')

    # custom legend order
    handles, labels = plt.gca().get_legend_handles_labels()
    ############## CHANGE ORDER HERE
    order = wdist_order
    plt.xticks(tick_places, ticknames)
    plt.legend([handles[i] for i in order],
               [labels[i] for i in order],
               loc=wd_loc)
    plt.show()

    # plotting the distances
    ## ACTIVATIONS
    # create the figure
    fig = plt.figure(figsize=(10, 5))

    # Get and plot the X sets
    xc = np.array(distances['activations'])
    mask = np.isfinite(xc)
    plt.scatter(np.array(range(len(xu)))[mask], xc[mask], marker='o', color='r', label=f'unclipped dist')  #### L0

    # clipped similarities
    xc = np.array(distances_clipped['activations'])
    mask = np.isfinite(xc)
    plt.scatter(np.array(range(len(xu)))[mask], xc[mask],
                marker='o',
                color='b',
                label=f'clipped dist')  #### LEGEND: 1

    # plot the baselines
    # indentical weights
    plt.hlines(act_id_dist, 0, len(xu) - 1, colors='orange', linestyles=':', label='identical weights')  #### L1
    # random init, same arch
    plt.hlines(np.mean(quantities['ad'][2]), 0, len(xu) - 1, colors='r', linestyles=':', label='random init')  #### L2

    plt.title(f'Activation Distance trajectories {titleadd}')
    plt.xlabel(x_title)
    plt.ylabel(f'Distance {ylabadd}')

    # custom legend order
    handles, labels = plt.gca().get_legend_handles_labels()
    ############## CHANGE ORDER HERE
    order = adist_order
    if ylog:
        plt.yscale('log')
    plt.xticks(tick_places, ticknames)
    plt.legend([handles[i] for i in order],
               [labels[i] for i in order],
               loc=ad_loc)
    # annotate with the formula for the distance
    # annotation_string=r'd $= TrC_1 + TrC_2 - 2||C_1^{1/2} C_2^{1/2}||_n$'
    # plt.annotate(annotation_string, xy=(11, 1.1*10**6))

    plt.show()


def plot_accuracy_trajectory(accuracies, acc_baseline, xticks=None, legend_loc='lower left', titleadd=''):
    fig = plt.figure(figsize=(10,5))

    plt.hlines(acc_baseline, 0, len(accuracies) - 1, label='unperturbed', color='g')
    plt.scatter(range(len(accuracies)), accuracies, label='perturbed models', color='m')
    plt.hlines(0.1, 0, len(accuracies)-1, label='chance', color='orange', linestyle=':')

    plt.legend(loc=legend_loc)
    xticklabs = xticks if xticks else list(range(len(accuracies)))
    plt.xticks(range(len(accuracies)), xticklabs)

    plt.ylim(0.05, 0.45)

    plt.xlabel('Perturbation')
    plt.ylabel('Test Accuracy')
    plt.title(f'Test Accuracy Trajectory {titleadd}')

    return


def plot_trace_nnorms(trace1s, trace2s, nnorms, xticks=None, legend_loc='lower_left'):
    fig = plt.figure(figsize=(10, 5))


    plt.scatter(range(len(trace1s)), trace1s)


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