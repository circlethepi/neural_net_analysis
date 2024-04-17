# imports
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

# specifics

# other packages/files


def find_cutoff_rank(spectrum, dims):#, var):
    """

    :param spectrum:
    :param dims:
    :return: the rank of the spectrum to force-match to the tail
    """
    gamma = dims[0]/dims[1]
    cut_val = (1 + np.sqrt(gamma))**2
    print(f'cut value: {cut_val}')

    rank = np.searchsorted(spectrum, cut_val)

    return rank * -1


def find_cutoff_layer(spec_hist, dims):#, var):
    """
    find the cutoff ranks for the spectrum history of one layer

    :param spec_hist:
    :param dims:
    :return:
    """
    layer_cutoffs = []
    for spec in spec_hist:
        rank = find_cutoff_rank(spec, dims)
        layer_cutoffs.append(rank)

    return layer_cutoffs


def find_cutoff_list(model_spectrum_history, dim_list):
    """

    :param model_spectrum_history:
    :param dim_list:
    :return: list of cutoff ranks for each layer's spectrum history
    """
    cutoff_ranks = []

    for spec_hist, dims in list(zip(model_spectrum_history, dim_list)):
        rank = find_cutoff_layer(spec_hist, dims)
        cutoff_ranks.append(rank)

    return cutoff_ranks
    # each element is a list of cutoffs for the model


def get_ranks(start_rank, end_rank, num_ints, scale='log'):
    if scale == 'log':
        start = np.log2(start_rank)
        end = np.log2(end_rank)
        rank_flt = np.logspace(start, end, num=num_ints, base=2)
    elif scale == 'lin':
        rank_flt = np.linspace(start_rank, end_rank, num=num_ints)
    else:
        raise Exception("need to choose log or lin scale")

    rank_int = [int(np.ceil(rank)) for rank in rank_flt]

    return rank_int


def get_model_ranks(rankslist, n_ints, scale='log'):
    cutoff_ranks = []
    for pair in rankslist:
        layer_ranks = get_ranks(pair[0], pair[1], n_ints, scale=scale)
        # print(layer_ranks)
        cutoff_ranks.append(layer_ranks)
    return cutoff_ranks


def match_spectrum_tails_regime(spectrum_analysis_obj, tail_match = 'mp', rankslist=None, spacescale='log'):
    """
    force-matches the tails of the trained spectrum to the init after the marchenko-pastur bound, which is found using
    the above functions.
    :param spectrum_analysis_obj:
    :param rankslist: list(2-tuple) where each 2-tuple is a start and end rank for each layer's spectrum
    :param spacescale: string 'lin' or 'log' for the spacing of the rankslists applied to the spec hist
    :return:
    """
    dim_list = spectrum_analysis_obj.layer_dims[:-1]
    spectrum_history = spectrum_analysis_obj.spectrum_history
    n_ints = len(spectrum_analysis_obj.epoch_history)

    if tail_match == 'mp':
        rank_bound_list = find_cutoff_list(spectrum_history, dim_list)
    elif tail_match == 'rankspace':
        if rankslist is None:
            raise Exception('to use rankspace rankfinding, need a rankslist')
        rank_bound_list = get_model_ranks(rankslist, n_ints, scale=spacescale)

    # calculating the init tail
    # and also the real tails
    # init_tail is a list of len(n_layers), where each entry is a list of len(epoch_history)
    init_tails = []
    lay_ep_tails = []
    scale_consts = []
    new_init_specs = []
    diffed_specs = []
    for i in range(len(spectrum_history)):
        layer_init_tails = []
        lay_tails = []
        lay_scales = []
        new_inits = []
        diffed_lay = []

        lay_init = spectrum_history[i][0]  # the init spectrum for the layer

        for j in range(1, len(spectrum_history[i])):    # starting AFTER the init
            # get the cutoff rank
            ep_rank = rank_bound_list[i][j-1]             # the cutoff rank
            # print(ep_rank)

            # calculate the init tail
            lay_init_sum = np.sum(lay_init[ep_rank:])   # getting the init sum
            layer_init_tails.append(lay_init_sum)       # add to the list

            # get the tail for the trained spectrum
            lay_spec = spectrum_history[i][j]           # the spectrum at this epoch
            lay_spec_sum = np.sum(lay_spec[ep_rank:])   # getting the epoch sum
            lay_tails.append(lay_spec_sum)              # add to the list

            # get the scaling to force-match the tails
            scale = lay_spec_sum / lay_init_sum         # get the scaling constant
            lay_scales.append(scale)                    # add to the list

            # calculate the rescaled init tail
            new = scale * lay_init.copy()               # scaled up init spectrum
            new_inits.append(new)                       # add to list

            # get the difference of the trained tail and the diffed tail
            diffed = np.subtract(lay_spec.copy(), new.copy())
            diffed = diffed * (diffed >= 0)             # making sure it's not terrible
            # diffed = [diffed[i] / lay_init[i] for i in range(len(lay_init))]
            diffed_lay.append(diffed)

        # add to the lists
        init_tails.append(layer_init_tails)
        lay_ep_tails.append(lay_tails)
        scale_consts.append(lay_scales)
        new_init_specs.append(new_inits)
        diffed_specs.append(diffed_lay)

    print('number of new spectrums', len(init_tails))
    print('WARNING: the first element of the diffed/normed spectra corresponds with the first checkpoint after init')
    return diffed_specs, new_init_specs, rank_bound_list


def eigen_expectation(spectrum):
    exp = 0
    density_spec = spectrum / np.sum(spectrum)
    for i in range(len(spectrum)):
        eigval = density_spec[i]
        exp += i * eigval

    return exp


def effective_dimensionality(spectrum_analysis_obj):
    """

    :param spectrum_analysis_obj: spectral_analysis.spectrum_analysis object
    :return: eff_dims: list(list(float)), a len(model.n_layers) list where each element is a len(model.ep_history)
    list containing the effective dimension for each checkpoint at which the spectrum was calculated during training
    """
    spectrum_history = spectrum_analysis_obj.spectrum_history
    normed_specs = spectrum_analysis_obj.normed_spectra

    # getting the effective dimensions for each layer
    eff_dims = []
    for i in range(len(spectrum_history)):
        layer_dims = []
        # get the init dimension
        init_spec = spectrum_history[i][0]
        init_dims = eigen_expectation(init_spec)
        layer_dims.append(init_dims)

        # get the ep history from normed specs
        for normed in normed_specs[i]:
            train_dims = eigen_expectation(normed)
            layer_dims.append(train_dims)

        eff_dims.append(layer_dims)

    return eff_dims
