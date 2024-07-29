import matplotlib
from matplotlib.lines import Line2D
import perturbation as pert
import spectral_analysis as spec
import alignment as align
import archive.class_splitter as cs
import neural_network as nn_mod
import network_similarity as sim
from utils import *

import numpy as np
import torch
import torchvision

import os
from tqdm import tqdm
#from tqdm.notebook import tqdm
import pickle
from matplotlib import pyplot as plt
from matplotlib import colors

from sklearn import manifold

# check if there is a GPU available
device = set_torch_device()

########################
### GLOBAL VARIABLES ###
########################
COMMON_SEED = 1234


class SinglePerturbationResultsConverter:
    """
    Converts the results from a single perturbation run into a 
    similaritiy matrix

    NOT USED - mostly use compute_pairwise_sims
    """
    def __init__(self, resultObj : pert.PerturbationResults, clipped=True):
        self.similarities_1 = resultObj.similarities_clipped if clipped\
              else resultObj.similarities
        self.clipped = clipped

        self.models = resultObj.models

        return
    


# Helper Functions
def compute_pairwise_sims(model_set, layer=1, w_clip=30, a_clip=64, 
                          similarity=True, labels=None, model_set2=None):
    """
    Compute the pairwise distances between a list of trained models for 
    a single layer

    :param model_set:   list(spec.spectrum_analysis) : models to compute 
    distances between
    :param layer    :   int :   the layer at which to compute the distances
    :param w_clip   :   int :   which rank to clip the weights to when doing 
                                the calculation
    :param a_clip   :   int :   which rank to clip the activations to when 
                                doing the calculation 
    :param similarity   :   bool    :   whether the metric should be similarity 
                                        (default True). If False, then the 
                                        metric is the BW2 distance for zero-
                                        mean gaussians
    """
    # move all models to gpu if available
    # for model in model_set:
    #     model.to(device)

    # if there are names, set those names. Otherwise, use generic model names
    # if labels:
    #     if len(labels) != len(model_set): # check that the number of given names is correct
    #         os.system('say "dummy do you even know how to count?"')
    #         raise Exception("""Number of models is not the same as number of 
    #                         names provided""")
    #     names = labels
    # else:
    #     names = [f'Model {i}' for i in range(len(model_set))]
    model_set2 = model_set2 if model_set2 else model_set
    

    # Results container
    pairwise_sims = {'activations' : [], 'weights' : []}

    # do the calculations
    metric = "similarities" if similarity else "distances"
    for i in range(len(model_set)):
        model1 = model_set[i]
        model_i_act = []
        model_i_way = []
        second_loop = range(i+1, len(model_set)) if model_set == model_set2 \
            else range(len(model_set2))
        for j in tqdm(second_loop, 
                      desc=f'Computing {i+1}th pairwise {metric}'):
            
            model2 = model_set2[j]
            # create the similarity object
            simobj = sim.network_comparison(model1, model2) 
                                            #names=(names[i], names[j]))

            # get the alignments
            set_seed(COMMON_SEED)
            simobj.compute_alignments(model1.train_loader, [layer])
            simobj.compute_cossim()

            # get the metrics
            activations, weights = simobj.network_distance(w_clip=w_clip, 
                                                           a_clip=a_clip, 
                                                           sim=similarity, 
                                                    return_quantities=False)
            
            model_i_act.append(activations[0])
            model_i_way.append(weights[0])
            
            del simobj
            del model2

        del model1
        print(model_i_act, model_i_way)

        pairwise_sims['activations'].append(model_i_act)
        pairwise_sims['weights'].append(model_i_way)
    
    print(pairwise_sims)
    
    # contstruct the similarity matrix
    if model_set == model_set2:
        act_sims = similarity_matrix_from_lists(pairwise_sims['activations'])
        way_sims = similarity_matrix_from_lists(pairwise_sims['weights']) 
    else:
        act_sims = np.array(pairwise_sims['activations'])
        way_sims = np.array(pairwise_sims['weights'])       

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
    """
    Computes a similarity matrix from lists of similarities of decreasing 
    length corresponding to the upper diagonal of a similarity matrix 

    :param lists:   list(list(float))   :   to create an nxn similarity matrix,
                                            this should be a list of n-1 lists
                                            decreasing in length from n-1 to 1
    """
    new_lists = []
    for l in lists:
        number_add = len(lists) - len(list(l))
        l_new = list(0 for _ in range(number_add)) + list(l)
        new_lists.append(l_new)
    
    similarity_matrix = np.array(new_lists) + np.eye(len(new_lists)) + \
                        np.array(new_lists).transpose()

    return similarity_matrix


def plot_similarity_matrix(sims, title, ticks=None, axis_label=None, 
                           split_inds=None, vrange=(0,1), rotation=0,
                           figsize=(10,10)):
    """
    plots a similarity matrix heatmap
    """
    fig = plt.figure(figsize=figsize)

    mask = np.triu(np.ones_like(sims, dtype=bool), k=0)
    
    mappy = plt.imshow(np.ma.array(sims, mask=mask), cmap='binary', 
                       vmin=vrange[0], vmax=vrange[1],
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

def compute_MDS(similarity_matrix, zero_index=None, pickle=None,):
    """
    Computes MDS projection using similarity matrix

    :param similarity_matrix:   array-like  :   similarity matrix to use 
    :param zero_index       :   int         :   index of element to center the 
                                                coordinates about. default None
    :param pickle           :   str         :   if desired, name of file to 
                                                save coords to as a pickled 
                                                variable. default None
    """

    # first, convert into a dissimilarity matrix
    dissims = np.ones(similarity_matrix.shape) - similarity_matrix

    # compute the MDS
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', eps=1e-16,
                       max_iter=1000, n_init=100, random_state=0)
    mds.fit_transform(dissims)
    coords = mds.embedding_

    if zero_index or isinstance(zero_index, int):
        coords -= coords[zero_index]
    
    if pickle:
        with open(f'{pickle}.pkl', 'wb') as file:
            pickle.dump(coords, file)

    return coords


def plot_MDS_coords(coords, n_models=None, labels=None, increments=None, 
                    text_locs=None, colors=None, legend_cols=2, 
                    legend_order=None, markers=None, accuracies=None, 
                    bar_range=(0,1), color_traj=None, steps=None, 
                    cb_norm=None, zero_incs=[0], figsize=(12, 10),
                    zero_sep=False, zero_lab=None, zero_color='red'):
    """
    *args are similarity matrices 

    color_traj is a colormap or none. If true, uses the default colormap
    increments is a bool whether or not to place the labels on the map (instead
    of placing them in the legend)
    epochs is a list of numbers corresponding to the epochs represented
    """
    color_map = plt.cm.plasma  

    if color_traj:
        assert steps is not None, "to use color_traj, you must include steps"
        # for i in range(len(steps)):
        #     if np.max(steps[i]) != 1: # normalize if necessary
        #         steps[i] /= np.max(steps[i])
        if cb_norm and 'log' not in cb_norm:
            lims_cb = (np.min([np.min(k) for k in steps]), np.max([np.max(k) for k in steps]))
        else:
            print('Using setting for bar_range since log scale being used')
            lims_cb = bar_range
        print(lims_cb)
        if accuracies is None:
            bar_range=lims_cb
    if color_traj == True:
        color_traj = color_map

    

    if n_models:
        split_indices = [0]+[sum(n_models[:i]) for i in range(1,len(n_models)+1)]
    else: 
        split_indices = [0, len(coords) -1]
    
    # print(split_indices)

    n_perturbations = len(n_models) if n_models else 1
    
     #check that increments are set OK
    # if increments and len(increments) != n_perturbations:
    #     raise Exception("""Increment list count and number of perturbation 
    #                     experiments represented must be the same""")
    
    colors = colors if colors is not None else plt.cm.viridis(np.linspace(0, 1, n_perturbations)) 
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


    # plotting the result
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)

    for i in range(n_perturbations):
        low = split_indices[i]
        hig = split_indices[i+1]
        xs = coords[low:hig, 0]
        ys = coords[low:hig, 1]
        #print(increments[i])
        
        if markers is None:
            mark = 'o'
        else:
            mark = markers[i] if markers[i] else 'o'
        # plotting the trajectory
        if zero_sep:
            plt.scatter(xs[0], ys[0], marker=mark, s=100, 
                        color=zero_color, zorder=5)
            if i == 0:
                zero_legend = [matplotlib.patches.Patch(facecolor=zero_color,
                                                        edgecolor=zero_color,
                                                        label=zero_lab)]
        if accuracies or color_traj:
            plt.plot(xs, ys, markersize=10, linestyle=':',
                 color='k', label=labels[i], linewidth=1, zorder=1)
            if accuracies:
                plt.scatter(xs, ys, c=accuracies[i], marker=mark, s=100,
                            cmap=color_map, vmin=bar_range[0], 
                            vmax=bar_range[1], zorder=2, norm=cb_norm)
                increment_color = color_map(accuracies[i][-1])
            if color_traj:
                step_i = np.array(steps[i])
                if cb_norm and 'log' in cb_norm:
                    colors[colors <= 0] = 1e10-4
                #print(colors)
                plt.scatter(xs, ys, c=step_i, cmap=color_traj, 
                            marker=mark, s=100, norm=cb_norm,
                            vmin=bar_range[0], vmax=bar_range[1], zorder=2)
                increment_color = color_traj(steps[i][-1])
        else:
            plt.plot(xs, ys, markersize=10, marker=mark, linestyle=':',
                 color=colors[i], label=labels[i], linewidth=1, zorder=1)
            increment_color = colors[i]
        
        # plotting the increments

        if increments and labels:
            # set the ith increments
            increments_i = ['']*(n_models[i]-1) + [f'{labels[i]}']
            #print(increments_i)
            if i in zero_incs:
                increments_i[0] = '0'
                if zero_sep:
                    increments_i[0] = f'{labels[i]}'
            for inc, x, y in zip(increments_i, xs, ys):
                #print(inc, x, y)
                if inc:
                    plt.annotate(inc, xy=(x, y), xytext=text_locs[i],
                                textcoords='offset points', 
                                ha='right', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.05', fc=increment_color, 
                                              alpha=0.2),
                                arrowprops=dict(arrowstyle='-', 
                                                    connectionstyle='arc3,rad=0'),
                                fontsize=16)
    
    if labels and not increments:
        if legend_order:
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend([handles[idx] for idx in legend_order],
                    [labels[idx] for idx in legend_order], loc='upper left', 
                    bbox_to_anchor=(0.1, -0.08), fontsize=16, ncol=legend_cols)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0.1, -0.08), fontsize=16, 
                ncol=legend_cols)
    if zero_lab:
        ax.legend(handles=zero_legend, fontsize=16)
    
    # setting the colorbar
    if accuracies or color_traj:
        # ticks
        cbar_ticks = steps[0] if color_traj else [np.min(np.array(accuracies)), np.max(np.array(accuracies))]
        cbar = plt.colorbar(ticks=cbar_ticks)
        cbar.ax.tick_params(labelsize=16)
        
        # define the tick labels
        def tick_format(x):
            if int(x) == x:
                y = int(x)
            elif 0.01 < x < 1e2:
                y = f'{x:.2f}'
            else:
                y = f'{x:.2e}'
            return y
        cbar_labels = [tick_format(k) for k in cbar_ticks]
        cbar.ax.set_yticklabels(cbar_labels)

        # label
        cbar_label = 'Accuracy' if accuracies else 'Epoch'
        cbar.set_label(cbar_label, fontsize=16)    
        
    plt.tick_params(axis='both', which='both', labelsize=16)
    plt.gca().set_aspect('equal')
    plt.show()
    
    return


def plot_compute_MDS(similarity_matrix, n_models=None, labels=None, 
                     increments=None, text_locs=None, colors=None,
                     legend_cols=2, legend_order=None, markers=None,
                     zero_index = None, accuracies=None, acc_range=(0,1)):
    """
    :param n_models: list-like for the the number of models for each 
    type of perturbation represented
    """
    color_map = plt.cm.plasma

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
    #print(coords)

    if zero_index is not None:
        coords -= coords[zero_index]
        #print(coords)
    #print(len(coords))

    # plotting the result
    fig = plt.figure(figsize=(12, 10))
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
        if accuracies:
            plt.plot(xs, ys, markersize=10, linestyle=':',
                 color='k', label=labels[i], linewidth=0.25, zorder=1)
            plt.scatter(xs, ys, c=accuracies[i], marker=mark, s=100,
                        cmap=color_map, vmin=acc_range[0], 
                        vmax=acc_range[1], zorder=2)
            increment_color = color_map(accuracies[i])
        else:
            plt.plot(xs, ys, markersize=10, linestyle=':',
                 color=colors[i], label=labels[i], linewidth=0.25, zorder=1)
            increment_color = colors[i]
        
        # plotting the increments

        if increments and labels:
            # set the ith increments
            increments_i = ['']*n_models[i] + [f'{labels[i]}']
            for inc, x, y in zip(increments[i], xs, ys):
                #print(inc, x, y)
                if inc:
                    plt.annotate(inc, xy=(x, y), xytext=text_locs[i],
                                textcoords='offset points', 
                                ha='right', va='bottom',
                                bbox=dict(boxstyle='round,pad=0.05', fc=increment_color, 
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
    plt.gca().set_aspect('equal')
    plt.show()

   # plt.savefig()

    return
