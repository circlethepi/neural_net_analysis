"""Package Import"""
import numpy as np

import alignment as align
import neural_network as nn_mod
import perturbation as pert
import spectral_analysis as spec
from utils import *

from tqdm import tqdm
#from tqdm.notebook import tqdm

"""
TO-DO

"""

class ActivationCollector():
    """
    Collects activations from a model
    """

    def __init__(self, path, epoch):
        """
        Takes in the path and epoch to find a model save file
        """
        specmod = spec.spectrum_analysis([], load=True, 
                                         path=path, epoch=epoch)
        
        # set the model
        self.model = specmod.model
        del specmod
        
        # set the activations
        self.activation_dict = None # [layer] -> [class] -> activations
        self.activation_loader = None # pert.Loader

        # self.
        
        return
    
    def get_all_activations(self, data_settings : pert.PerturbationSettings = \
                            pert.default_perturbation_settings):
        """
        Gets the Activation Dict and sets the attribute
        activation_dict: [layer] -> [class] -> activations
        """
        
        activation_dict, loader = get_model_class_acts_all_inds_all_layers(self.model, 
                                                                   data_settings)
        self.activation_dict = activation_dict
        self.activation_loader = loader

        return activation_dict, loader


# Helper Functions
def get_model_class_acts(model, dataloader,#: pert.PerturbationSettings, 
                         class_ind, layer : int):
    """
    Collects all the activations from a given model at the given layer
    """
    # get the activations
    get_acts = lambda *args: [align.space_to_batch(act) for act in \
                              align.get_activations(*args)]
    
    acts = []
    for x, y in tqdm(dataloader, desc="Computing Activations"):
        # mask to get just the points we want 
        mask = (y == class_ind)
        x = x[mask]
        x = x.to(device)
        act1 = get_acts(x, [layer], model)
        acts.append(act1[0])
    
    # each collection of activations is n x d where
        # n = number of activations, d = dimension

    return torch.cat(acts, dim=0)


def get_model_class_acts_multi_ind(model, dataloader, class_inds, layer : int):
    # get the activations for each class index
    activations = []
    for i in tqdm(class_inds, "Getting Class Activations"):
        activations.append(get_model_class_acts(model, dataloader, i, layer))
    
    result = dict(zip(class_inds, activations))

    return result


def get_model_class_acts_all_inds_all_layers(model, data_set : pert.PerturbationSettings):
    unmod = getattr(data_set, 'unmod_classes')
    modcl = getattr(data_set, 'mod_classes')
    class_inds = []
    for t in (unmod, modcl):
        if t is not None:
            class_inds += t
    #print(class_inds)
    #class_inds = data_set.unmod_classes + data_set.mod_classes
    # load the data
    dataloader = pert.subset_class_loader(data_set)[0]

    layers = list(range(1, len(model.layers)))
    #print(layers)

    layer_acts = []
    for k in layers:
        layer_acts.append(get_model_class_acts_multi_ind(model, dataloader, 
                                                         class_inds, k))
    
    result = dict(zip(layers, layer_acts))

    return result, dataloader



class ActivationCollection():
    """
    A collection of ActivationCollectors, one for each epoch
    """
    def __init__(self, path, epochs, data_set : pert.PerturbationSettings):
        """
        Takes in the path and a list of epochs a calculates the activations for
        each class at each layer
        """
        # for each epoch, create an Activation Collector
        collect = []
        for ep in epochs:
            c = ActivationCollector(path, ep)
            acts, load = c.get_all_activations(data_set)
            collect.append(acts)
        # collect has all dicts
        self.activations = dict(zip(epochs, collect))
        self.loader = load
        # activations: [epoch] -> [layer] -> [class]

        return


class ActivationPlotter():
    """
    Takes an Activation Collection and computes the coordinates for the 
    activation clouds 

    Also plots them 
    """

    def __init__(self, collection : ActivationCollection):
        
        self.activations = reshape_activation_dictionary(collection.activations)
        self.loader = collection.loader

        self.layers = list(self.activations.keys())
        self.epochs = list(self.activations[self.layers[0]].keys())
        self.classes = list(self.activations[self.layers[0]][self.epochs[0]].keys())
        # each collection of activations is n x d where
        # n = number of activations, d = dimension

        self.coordinates = None
        # will be 
        # coordinates: [epoch] -> [layer] -> [class] -> coordinates in 2D

        return


def reshape_activation_dictionary(activation_dict):
    """
    Reshape from [epoch] -> [layer] -> [class] -> activations 
    to           [layer] -> [epoch] -> [class] -> activations
    """
    # epochs = list(activations_dict.keys())
    # layers = list(activations_dict[epochs[0]].keys())

    transformed_dict = {}

    for epoch, epoch_data in activation_dict.items():
        for layer, class_data in epoch_data.items():
            if layer not in transformed_dict:
                transformed_dict[layer] = {}
            for class_name, data in class_data.items():
                if epoch not in transformed_dict[layer]:
                    transformed_dict[layer][epoch] = {}
                transformed_dict[layer][epoch][class_name] = data


    return transformed_dict


def collect_epoch_activations(activation_dict_re, layer, epoch):
    """
    gets all the coordinates from a reshaped activations dict
    ([layer] -> [epoch] -> [class] -> activations)
    """
    assert layer in activation_dict_re.keys(), 'Invalid layer selection'
    assert epoch in activation_dict_re[layer].keys(), 'Invalid epoch selection'

    all_class_acts = activation_dict_re[layer][epoch]



    return