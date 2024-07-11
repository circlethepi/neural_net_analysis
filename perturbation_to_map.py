import perturbation as pert
import spectral_analysis as spec
import alignment as align
import class_splitter as cs
import neural_network as nn_mod
import network_similarity as sim

import numpy as np
import torch
import torchvision

import os
from tqdm import tqdm
from tqdm.notebook import tqdm
import pickle
from matplotlib import pyplot as plt
from matplotlib import colors

from sklearn import manifold

########################
### GLOBAL VARIABLES ###
########################
COMMON_SEED = 1234


class SinglePerturbationResultsConverter:
    """
    Converts the results from a single perturbation run into a 
    similaritiy matrix
    """
    def __init__(self, resultObj : pert.PerturbationResults, clipped=True):
        self.similarities_1 = resultObj.similarities_clipped if clipped\
              else resultObj.similarities
        self.clipped = clipped

        self.models = resultObj.models

        return
    


# Helper Functions
def compute_pairwise_sims(model_set, layer=1, w_clip=30, a_clip=64, 
                          similarity=True, labels=None):
    """
    Compute the pairwise distances between a list of trained models for 
    a single layer
    :param model_set:   list(spec.spectrum_analysis) : models to compute 
    distances between

    """
    # if there are names, set those names. Otherwise, use generic model names
    if labels:
        if len(labels) != len(model_set): # check that the number of given names is correct
            os.system('say "dummy do you even know how to count?"')
            raise Exception("""Number of models is not the same as number of 
                            names provided""")
        names = labels
    else:
        names = [f'Model {i}' for i in range(len(model_set))]

    # Results container
    pairwise_sims = {'activations' : [], 'weights' : []}

    # do the calculations
    metric = "similarities" if similarity else "distances"
    for i in range(len(model_set)):
        model1 = model_set[i]
        model_i_act = []
        model_i_way = []
        for j in tqdm(range(i+1, len(model_set)), 
                      desc=f'Computing {i+1}th pairwise similarity'):
            
            model2 = model_set[j]
            # create the similarity object
            simobj = sim.network_comparison(model1, model2, 
                                            names=(names[i], names[j]))

            # get the alignments
            cs.set_seed(COMMON_SEED)
            simobj.compute_alignments(model1.train_loader, [layer])
            simobj.compute_cossim()

            # get the metrics
            activations, weights = simobj.network_distance(w_clip=w_clip, 
                                                           a_clip=a_clip, 
                                                           sim=similarity, 
                                                    return_quantities=False)
            
            model_i_act.append(activations[0])
            model_i_way.append(weights[0])
        print(model_i_act, model_i_way)

        pairwise_sims['activations'].append(model_i_act)
        pairwise_sims['weights'].append(model_i_way)
    
    print(pairwise_sims)
    
    # contstruct the similarity matrix
    act_sims = similarity_matrix_from_lists(pairwise_sims['activations'])
    way_sims = similarity_matrix_from_lists(pairwise_sims['weights'])        

    return act_sims, way_sims


def compute_similarity_line(perturbation_result_similarities):

    pairwise = []
    for i in range(len(perturbation_result_similarities)):
        sim_i = perturbation_result_similarities[i]
        model_i = [1- (sim_i - perturbation_result_similarities[j]) for j in 
                   range(i+1, len(perturbation_result_similarities))]
        print(model_i)
        pairwise.append(model_i)
    
    # construct the similarity matrix
    sims = similarity_matrix_from_lists(pairwise) 
            
    return sims


def similarity_matrix_from_lists(lists):
    new_lists = []
    for l in lists:
        print()
        number_add = len(lists) - len(list(l))
        l_new = list(0 for _ in range(number_add)) + list(l)
        new_lists.append(l_new)
    
    similarity_matrix = np.array(new_lists) + np.eye(len(new_lists)) + \
                        np.array(new_lists).transpose()

    return similarity_matrix

def plot_similarity_matrix(sims, title, ticks=None, axis_label=None, 
                           split_inds=None, vrange=(0,1), rotation=0):
    fig = plt.figure(figsize=(10, 10))

    mask = np.triu(np.ones_like(sims, dtype=bool))
    
    mappy = plt.imshow(np.ma.array(sims, mask=mask), cmap='binary', vmin=vrange[0], vmax=vrange[1],
                        interpolation='nearest')

    plt.colorbar(mappy, fraction=0.045)
    #cbar.tickparams(labelsize=pert.axis_fontsize)


    if split_inds:
        for ind in split_inds:
            xs = [-0.5, len(sims)-0.5]
            ys = [ind-0.5, ind-0.5]
            plt.plot(xs, ys, color='r')
            plt.plot(ys, xs, color='r')
    

    if ticks:
        plt.xticks(list(range(len(sims))), ticks)
        plt.yticks(list(range(len(sims))), ticks)
    if axis_label:
        plt.xlabel(axis_label, fontsize=pert.axis_fontsize)
        plt.ylabel(axis_label, fontsize=pert.axis_fontsize)

    plt.title(title, fontsize=pert.axis_fontsize)
    plt.tick_params(axis='both', which='both', labelsize=pert.axis_fontsize)
    plt.tick_params(axis='x', labelrotation=rotation)

    plt.show()

    return


def plot_compute_MDS(similarity_matrix, n_models=None, labels=None, 
                     increments=None, text_locs=None, colors=None,
                     legend_cols=2, legend_order=None, markers=None,
                     zero_index = None, accuracies=None):
    """
    :param n_models: list-like for the the number of models for each 
    type of perturbation represented
    """
    # first, convert into a dissimilarity matrix
    dissims = np.ones(similarity_matrix.shape) - similarity_matrix

    # get where to finish the 
    if n_models:
        split_indices = [0]+[sum(n_models[:i]) for i in range(1,len(n_models)+1)]
    else: 
        split_indices = [0, len(dissims) -1]
    #print(split_indices)

    n_perturbations = len(n_models) if n_models else 1

    #check that increments are set OK
    if len(increments) != n_perturbations:
        raise Exception("""Increment list count and number of perturbation 
                        experiments represented must be the same""")

    colors = colors if colors else plt.cm.viridis(np.linspace(0, 1, n_perturbations)) 
    if labels:
        labels = [labels[i] if labels[i] else "" for i in range(len(labels))]
        make_legend = True
    else:
        labels = ["" for i in range(n_perturbations)] 
        make_legend = False
    #labels = labels if labels else None#[f'Perturbation {i+1}' for i in 
                                   # range(n_perturbations)]
    text_locs = text_locs if text_locs else [(-12,-12) for i in range(len(labels))]
    #print(text_locs)

    # compute the MDS
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', eps=1e-16,
                       max_iter=1000, n_init=100)
    mds.fit_transform(dissims)
    coords = mds.embedding_
    print(coords)

    if zero_index is not None:
        coords -= coords[zero_index]
        print(coords)
    #print(len(coords))

    # plotting the result
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)

    for i in range(n_perturbations):
        low = split_indices[i]
        hig = split_indices[i+1]
        xs = coords[low:hig, 0]
        ys = coords[low:hig, 1]
        print(increments[i])
        
        if not markers:
            mark = 'o'
        else:
            mark = markers[i] if markers[i] else 'o'
        # plotting the trajectory
        plt.plot(xs, ys, marker=mark, markersize=10,
                 color=colors[i], label=labels[i], linewidth=0.25)
        if accuracies:
            plt.scatter(xs, ys, c=accuracies[i], marker=mark, markersize=10,
                        cmap=plt.cm.plasma, vmin=0, vmax=1)
        
        # plotting the increments
        if increments:
            if increments[i]:
                for inc, x, y in zip(increments[i], xs, ys):
                    #print(inc, x, y)
                    if inc:
                        plt.annotate(inc, xy=(x, y), xytext=text_locs[i],
                                    textcoords='offset points', 
                                    ha='right', va='bottom',
                                    bbox=dict(boxstyle='round,pad=0.05', 
                                              fc=colors[i], 
                                              alpha=0.2),
                                    arrowprops=dict(arrowstyle='-', 
                                                    connectionstyle='arc3,rad=0'),
                                    fontsize=16)
    
    if make_legend:
        if legend_order:
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend([handles[idx] for idx in legend_order],
                    [labels[idx] for idx in legend_order], loc='upper left', 
                    bbox_to_anchor=(0.1, -0.08), fontsize=16, ncol=legend_cols)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0.1, -0.08), fontsize=16, 
                ncol=legend_cols)
    if accuracies:
        plt.colorbar()
        
    plt.tick_params(axis='both', which='both', labelsize=16)
    plt.show()

    return
