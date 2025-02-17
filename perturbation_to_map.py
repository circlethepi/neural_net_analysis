from re import L
import matplotlib
from matplotlib.lines import Line2D
import perturbation as pert
import spectral_analysis as spec
import alignment as align
#import archive.class_splitter as cs
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
import json
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import copy

from sklearn import manifold
from typing import Union, Optional

# check if there is a GPU available
device = set_torch_device()

########################
### GLOBAL VARIABLES ###
########################
COMMON_SEED = 1234
GLOBAL_COMPARISON_DICT = '../../../datascope/menard/group/mohata1/comparison_library'

# Helper Functions
def compute_pairwise_sims(model_set, dataloader=None, layer=1, w_clip=30, a_clip=64, 
                          similarity=True, labels=None, model_set2=None, align=True):
    """
    Compute the pairwise distances between a list of trained models for 
    a SINGLE layer which is indicated

    :param model_set:   list(spec.spectrum_analysis) : models to compute 
    distances between. These should have set .train_loader features where the
    train_loaders are the loaders used to calculate activations for the 
    corresponding model.
    :param dataloader   :   torch.dataloader    :   dataloader for alignments
    :param layer    :   int :   the layer at which to compute the distances
    :param w_clip   :   int :   which rank to clip the weights to when doing 
                                the calculation
    :param a_clip   :   int :   which rank to clip the activations to when 
                                doing the calculation 
    :param similarity   :   bool    :   whether the metric should be similarity 
                                        (default True). If False, then the 
                                        metric is the BW2 distance for zero-
                                        mean gaussians
    :param model_set2   :   list(spec.spectrum_analysis)    :   if this is set,
    this indicates the calculation of a rectangular/asymmetric similarity 
    matrix, for use (for example) in calculating a similarity matrix by chunks
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

    # set the dataloaders
    if dataloader is None:
        dataloader = model_set[0].train_loader

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
            simobj = sim.NetworkComparison(model1, model2) 
                                            # names=(names[i], names[j]))

            # get the alignments
            #set_seed(COMMON_SEED)
            #simobj.compute_alignments(model1.train_loader, [layer])
            #if align:
            if layer == 1:
                simobj.compute_alignments(dataloader, [layer])
            else:
                simobj.compute_alignments(dataloader, [layer-1, layer])
            #simobj.compute_cossim()

            # get the metrics
            activations, weights = simobj.network_distance(w_clip=w_clip, 
                                                           a_clip=a_clip, 
                                                           sim=similarity,
                                                           layers=[layer], 
                                                    return_quantities=False,
                                                    align=align)
            
            model_i_act.append(activations[-1])
            model_i_way.append(weights[-1])
            
            del simobj
            del model2

        del model1
        #print(model_i_act, model_i_way)

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

    return act_sims, way_sims#, r2s


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
                           figsize=(10,10), save=False, #saveloc='../image_hold',
                           savepath='img', nan_color='midnightblue',
                           zero_color='#efefef',
                           split_color='r'):
    """
    plots a similarity matrix heatmap
    """
    fig = plt.figure(figsize=figsize)

    mask = np.triu(np.ones_like(sims, dtype=bool), k=0)
    nan_sims = np.isnan(sims)
    nan_sims = np.ma.masked_where(nan_sims==False, nan_sims)
    nan_sims = np.ma.array(nan_sims, mask=mask)
    # colormap for normal sims
    cmap = copy.copy(matplotlib.cm.binary)
    cmap.set_bad(zero_color)
    # colormap for nan values
    cnan = matplotlib.colors.ListedColormap([nan_color,nan_color])

    #plt.imshow(np.ma.array(masked_sims, mask=mask), cmap='BuGn_r')
    #plt.imshow(np.ma.array(sims, mask=mask), cmap='BuGn')
    mappy = plt.imshow(np.ma.array(sims, mask=mask), cmap=cmap, 
                       vmin=vrange[0], vmax=vrange[1],
                        interpolation='nearest')
    plt.imshow(nan_sims, aspect='auto', cmap=cnan, vmin=0, vmax=1)

   
    

    cbar = plt.colorbar(mappy, fraction=0.045)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)


    if split_inds:
        for ind in split_inds:
            xs = [-0.5, len(sims)-0.5]
            ys = [ind-0.5, ind-0.5]
            plt.plot(xs, ys, color=split_color)
            plt.plot(ys, xs, color=split_color)
    

    if ticks:
        plt.xticks(list(range(len(sims))), ticks)
        plt.yticks(list(range(len(sims))), ticks)
    if axis_label:
        plt.xlabel(axis_label, fontsize=pert.axis_fontsize)
        plt.ylabel(axis_label, fontsize=pert.axis_fontsize)

    plt.title(title, fontsize=pert.axis_fontsize)
    plt.tick_params(axis='both', which='both', labelsize=pert.axis_fontsize)
    plt.tick_params(axis='x', labelrotation=rotation)
    
    if save:
        #savepath = f'{saveloc}/MAT{title}'
        plt.savefig(savepath, bbox_inches='tight')
    
    plt.show()

    return


def compute_MDS(similarity_matrix, zero_index=None, pickle=None, 
                align_coords=True, yflip=False, xflip=False):
    """
    Computes MDS projection using similarity matrix

    :param similarity_matrix:   array-like  :   similarity matrix to use 
    :param zero_index       :   int         :   index of element to center the 
                                                coordinates about. default None
    :param pickle           :   str         :   if desired, name of file to 
                                                save coords to as a pickled 
                                                variable. default None
    """
    metric = True # whether to do metric MDS
    # check if any of the entries are NaN
    # if there are, replace them with ones so that the dissimilarity matrix
    # treats them as missing values
    if np.any(np.isnan(similarity_matrix)):
        metric = False
        similarity_matrix = np.nan_to_num(similarity_matrix, nan=1)
        # replace the diagonal with a small value since 0 will be the same as missing
        np.fill_diagonal(similarity_matrix, 1+np.finfo(float).eps)
    # first, convert into a dissimilarity matrix
    dissims = np.ones(similarity_matrix.shape) - similarity_matrix

    # compute the MDS
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', eps=1e-16,
                       max_iter=1000, n_init=100, random_state=0, metric=metric,
                       normalized_stress=False)
    mds.fit_transform(dissims)
    coords = mds.embedding_

    if zero_index:
        assert isinstance(zero_index, int), "Invalid datatype for zero index"
        coords -= coords[zero_index]
    
    if align_coords:
        coords = align_traj_to_x(coords, yflip=yflip, xflip=xflip, 
                                 zero_ind=zero_index)

    if pickle:
        with open(f'{pickle}.pkl', 'wb') as file:
            pickle.dump(coords, file)

    return coords


def plot_MDS_coords(coords, title, n_models=None, labels=None, increments=None, 
                    text_locs=None, colors=None, legend_cols=2, 
                    legend_order=None, markers=None, accuracies=None, 
                    bar_range=(0,1), color_traj=None, steps=None, 
                    cb_norm=None, zero_incs=[0], figsize=(12, 10),
                    zero_sep=False, zero_lab=None, zero_color='red', 
                    align_coords=True, yflip=False, xflip=False, zero_ind=None,
                    save=False, saveloc='../image_hold',
                    xlim=None, ylim=None, xrot=0, show=True, colorbar=False):
    """
    *args are similarity matrices 

    color_traj is a colormap, True, or None. If True, uses the default colormap
    increments is a bool whether or not to place the labels on the map (instead
    of placing them in the legend)
    epochs is a list of numbers corresponding to the epochs represented
    zero_lab is what to label the zero point if zero_sep is true
    zero_sep is whether to color the zeroth point separately from everything else
    zero_color is the color of the zeroth point

    zero_incs 
    """
    # align the coordinates if true and flip if true
    if align_coords:
        coords = align_traj_to_x(coords, yflip=yflip, xflip=xflip, 
                                 zero_ind=zero_ind)

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

    # plot each curve :)
    for i in range(n_perturbations):
        low = split_indices[i]
        hig = split_indices[i+1]
        xs = coords[low:hig, 0]
        ys = coords[low:hig, 1]
        #print(increments[i])
        
        if markers is None:
            mark = 'o'
        else:
            if i < len(markers):
                mark = markers[i] if markers[i] else 'o'
            else:
                mark = 'o'
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
                increment_color = color_traj(step_i[-1])
        else:
            plt.plot(xs, ys, markersize=10, marker=mark, linestyle=':',
                 color=colors[i], label=labels[i], linewidth=1, zorder=1,
                 mew=0)
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
                                bbox=dict(boxstyle='round,pad=0.05', 
                                          fc=increment_color, alpha=0.2),
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
    if (accuracies or color_traj) and colorbar:
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
    
    # setting the axis limits if they are given:
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.tick_params(axis='both', which='both', labelsize=16)
    plt.tick_params(axis='x', labelrotation=xrot)
    plt.title(title, fontsize=16)
    plt.gca().set_aspect('equal')
    

    plt.show()

    if save:
        savepath = f'{saveloc}/MDS{title}_var'
        plt.savefig(savepath, bbox_inches='tight')
    
    return 


def align_trajectory(trajectory1, trajectory2):
    """
    Aligns trajectory 2 to trajectory 1 using SVD alignment matrix process
    Trajectories are coordinates calculated from pairwise similarities of model
    checkpoints taken over the course of training then MDS projected into 2D

    each tajectory should be np.arrays of shape (n_checkpoints, 2)
    """
    # get the alignment
    cross_cov = trajectory1.T @ trajectory2
    u, s, vh = np.linalg.svd(cross_cov)
    align = u @ vh

    # align the coordinates
    new_trajectory2 = np.array([align @ c.T for c in trajectory2])

    return new_trajectory2#, align


def align_trajectory_collection(*trajectories, ref=None):
    if ref is None:
        ref = trajectories[0]
    
    new_trajectories = []
    #alignments = []
    for traj in trajectories:
        #new, align = align_trajectory(ref, traj)
        new = align_trajectory(ref, traj)
        new_trajectories.append(new)
        #alignments.append(align)

    return new_trajectories#, alignments


def align_traj_to_x(trajectory, yflip=False, xflip=False, zero_ind=None):
    """
    Trajectory is array, has shape (n, 2)
    """

    # find PCA of coordinates
    #trajectory -= np.mean(trajectory)
    cov = trajectory.T @ trajectory
    vals, vecs = np.linalg.eigh(cov)
    np.flip(vals, axis=-1)
    np.flip(vecs, axis=-1)

    # find angle between x axis and PC1
    pc1 = vecs[:, 0]
    # get angle
    t = np.arctan2(pc1[1], pc1[0])
    # rotation matrix
    rot = make_2d_rotation(-t)

    # rotate the coordinates
    new_traj = np.array([rot@k for k in trajectory])

    # flip if necessary
    flipmaty = np.array([[-1, 0], [0, 1]])
    flipmatx = np.array([[1, 0], [0, -1]])
    if yflip:
        new_traj = np.array([flipmaty@k for k in new_traj])
    if xflip:
        new_traj = np.array([flipmatx@k for k in new_traj])
    if zero_ind is not None:
        new_traj -= new_traj[zero_ind]

    return new_traj#, vals, vecs


def make_2d_rotation(t):

    mat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) 

    return mat


def get_variance_axes(coordinates):
    """

    :param coordinates : list   - list of aligned coordinates
    each set of coordinates should be of shape (2, k)

    :return var_plot_info : dict - dict of all the info needed for a 2d
                                   MDS plot with variances
    """
    for c in coordinates:
        assert c.shape == coordinates[0].shape, "all coordinates must have the same shape"
    #print(coordinates[0].shape)
    # collect the corresponding coordinates for each run
    collections = []
    for k in range(coordinates[0].shape[0]):
        collect = np.array([ c[k] for c in coordinates ])
        collections.append(collect)
        #print(collect)

    # get the means
    xs = [np.mean(c[:,0]) for c in collections]
    ys = [np.mean(c[:,1]) for c in collections]


    # do PCA on the collections of corresponding checkpoints
    u1 = [] # first PC first component
    v1 = [] #   "    " second     "
    u2 = [] # second PC first component
    v2 = [] #   "    "  second    "

    s1 = [] # sqrt first eigval
    s2 = [] # sqrt second eigval

    # get the variances :)
    for i in range(len(collections)):
        coll = collections[i]
        #coll_mean = np.array([xs[i], ys[i]])
        coll_mean = np.mean(coll, axis=0)
        coll -= coll_mean
        # print('mean-centered coordinates')
        # print(coll)
        cov = (1/coll.shape[0]) * coll.T @ coll # (2, 2)

        # print('covariance')
        # print(cov)
        vals, vecs = np.linalg.eigh(cov)
        vals, vecs = np.flip(vals), np.flip(vecs, axis=1)

        # print('eigenvalues : ', vals)
        # print('SD          : ', np.sqrt(vals))

        # append to the correct list
        # directions
        u1.append(vecs[0,0])
        v1.append(vecs[1,0])
        u2.append(vecs[0,1])
        v2.append(vecs[1,1])
        # lengths
        s1.append(np.sqrt(vals[0]))
        s2.append(np.sqrt(vals[1]))

    label = ['xs', 'ys', 'u1', 'v1', 'u2', 'v2', 's1', 's2']
    quants = (xs, ys, u1, v1, u2, v2, s1, s2)
    var_plot_info = dict(zip(label, quants))

    return var_plot_info


class VariancePlot:

    def __init__(self, *args, split=None):
        """
        :param  *args : array-like - set of coordinates  
        :param  split : number of coordinates per trajectory if not split
        """
        # if split, chop up the coordinates into the appropriate splitting
        if split:
            # make sure getting only one set of coordinates
            all = np.squeeze(np.array(args))
            #print(all.shape)
            # make sure it is the proper shape, and the number of coords per
            # instance given is valid
            assert all.shape[1] == 2
            assert all.shape[0] % split == 0

            args = []
            for k in range(all.shape[0] // split):
                args.append(all[k*split: (k+1)*split])

        # # first, align all the coordinates to x
        # coords = [align_traj_to_x(c, zero_ind=0) for c in args]
        # # print(len(coords))
        # # then align all coordinates to 1 reference
        # aligned_coords = align_trajectory_collection(*coords)
        # # print(len(aligned_coords))

        self.coordinates = args #aligned_coords

        # other features
        self.plot_info = None

        return

    def get_variance_plot_info(self, mean0=True, mean_ind=0):
        plot_info = get_variance_axes(self.coordinates)
        self.plot_info = plot_info
        if mean0:
            _ = self.set_mean_point_origin(ind=mean_ind)
        return plot_info
    
    def set_mean_point_origin(self, ind=0):
        """
        Translates all coordinates so that the first mean coordinate is (0, 0)
        (or whichever index in the trajectory you desire to be at the origin)
        (but by default this is the first one)
        """
        # get the corresponding values in the mean trajectory
        xmean = self.plot_info['xs'][ind]
        ymean = self.plot_info['ys'][ind]
        first_mean = np.array([xmean, ymean])

        # first, update all the means in plot_info
        self.plot_info['xs'] = [c-xmean for c in self.plot_info['xs']]
        self.plot_info['ys'] = [c-ymean for c in self.plot_info['ys']]

        # now, update all the coordinates
        self.coordinates -= first_mean

        return self.coordinates, self.plot_info

    def plot_variance(self, title=None, variance=True, ylog=False, xlog=False,
                      ticks=None, ticklabs=None, xlab='Step', xrot=0,
                      save=False, savename='img'):
        """
        PLot the change in variance over the course of the trajectories

        """
        fig = plt.figure(figsize=(10,5))

        if variance:
            pc1 = [k**2 for k in self.plot_info['s1']]
            pc2 = [k**2 for k in self.plot_info['s2']]
        else:
            pc1 = self.plot_info['s1']
            pc2 = self.plot_info['s2']
        
        if ticks is None:
            ticks = list(range(len(pc2)))

        plt.plot(ticks, pc1, label='PC1', marker='o', linewidth=0.5)
        plt.plot(ticks, pc2, label='PC2', marker='o', linewidth=0.5)

        if title is None:
            if variance:
                title = 'Variance over Trajectory'
            else:
                title = 'Standard Deviation over Trajectory'
        plt.title(title, fontsize=16)
        plt.xlabel(xlab, fontsize=16)

        ylab = 'Variance' if variance else 'Standard Deviation'
        plt.ylabel(ylab, fontsize=16)
        if ylog:
            plt.yscale('log')
        if xlog:
            plt.xscale('log')
        plt.tick_params(axis='both', which='both', labelsize=16)


        if ticklabs is None:
            ticklabs = ticks
        plt.xticks(ticks, ticklabs)

        plt.legend(fontsize=16)
        plt.tick_params(axis='x', labelrotation=xrot)

        if save:
            plt.savefig(savename, bbox_inches='tight')

        plt.show()

        return

    def plot_ellipse_error(self, title='', sd_mult=1,
                          ticks=None, ticklabs=None, xlab=None, xlog=False,
                          ylog=False, save=False, savename='img', xrot=0,
                          plot=True, ylim=None, xlim=None):
        """
        Plot the area of the ellipse over the course of the trajectory
        """
        if self.plot_info is None:
            _ = self.get_variance_plot_info
        
        # get the area of each ellipse defined by the SDs 
        ## first collect the SDs assc. w each PC 
        sd1 = self.plot_info['s1']
        sd2 = self.plot_info['s2']

        ## calculate area
        areas = [sd1[k]*sd2[k]*np.pi*sd_mult**2 for k in range(len(sd1))]

        if plot:
            ## plot
            fig = plt.figure(figsize=(10,5))

            if ticks is None:
                ticks = list(range(len(sd1)))
            if ticklabs is None:
                ticklabs = ticks

            if xlog and 0 in ticks:
                ticks = ticks[1:]
                areas = areas[1:]
                ticklabs = ticklabs[1:]
            plt.plot(ticks, areas, label='error', marker='o', linewidth=0.5)

            title += f' $\\pm${sd_mult}$\\sigma$'
            if xlab is None:
                xlab = 'Step'
                
            plt.title(title, fontsize=16)
            plt.xlabel(xlab, fontsize=16)

            ylab = 'error'
            plt.ylabel(ylab, fontsize=16)
            if ylog:
                plt.yscale('log')
            if xlog:
                plt.xscale('log')
            
            if ylim is not None:
                plt.ylim(ylim[0], ylim[1])
            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])
            plt.tick_params(axis='both', which='both', labelsize=16)

            plt.xticks(ticks, ticklabs)

            plt.legend(fontsize=16)
            plt.tick_params(axis='x', labelrotation=xrot)

            if save:
                plt.savefig(savename, bbox_inches='tight')

            plt.show()

        return areas    


def plot_variance_plot(coordinates, plot_info=None, title="Variance Plot", sd_mult=2,
                       n_models=None, labels=None, increments=None, inc_ind=-1,
                       inc_fontsize=None, fontsize=20,
                       text_locs=None, colors=None, legend_cols=2, 
                       legend_order=None, markers=None, markersize=5,
                       legend_loc='best',

                       accuracies=None, 
                       bar_range=(0,1), color_traj=None, steps=None, traj_a=1,
                       cb_norm=None, colorbar=False, color_list=None,
                       zero_incs=[0], figsize=(12, 10), 
                       zero_sep=False, zero_lab=None, zero_color='red', 

                       align_coords=True, yflip=False, xflip=False, zero_ind=None,

                       #save=False, saveloc='../image_hold', 
                       xlim=None, ylim=None, xrot=0, 
                       
                       varwidth=1e-3, 

                       mean_color='k', varcols=('#757575', '#757575'),
                       mean_symbol='D', mean_mark_size=5,
                       mean_lab='Mean Trajectory',

                       save=False, savename='img',

    ):
    """
    --- Main Needs ---
    :param  coordinates - np.array of size nx2 of coordinates to plot
    :param  plot_info - output from get_variance_axes
    :param  title - str title of the plot
    :param  n_models - the number of models/points for each trajectory (since 
            all coordinates are passed in at once, this splits them)
    
    --- Additional Coordinate Manipulation ---
    :param  align_coords - bool on whether of align the coords (if not aligned)
    :param  xflip - bool whether to flip the coordinates over x-axis
    :param  yflip - bool whether to flip the coordinates over y-axis
    :param  zero_ind - if aligning, which coordinate to center about 0

    --- Gradient / Colormap Points ---
    :param  color_traj - a colormap, True, or None. If True, uses the default 
            colormap. If not None, then the points are plotted according to the 
            colormap according to the values in steps (usually epoch values or 
            accuracies)
    :param  color_list - None (default) or a list of lists each with RGBA values
            for the corresponding points
    :param  steps - list of lists with values for color_traj. usually epochs or
            model accuracies
    :param  accuracies - list of lists with accuracy values for color_traj.
            indicates explicitly that these are accuracies
    :param  traj_a - float in (0,1) for alpha value of each trajectory's color 
    
    --- Colorbar ---
    :param  cb_norm - str norm for the colorbar
    :param  colorbar - whether to show colorbar
    :param  bar_range - 2-tuple of hi-lo values to show on the colorbar

    --- Trajectory Distinction ---
    :param  increments - Bool whether or not to place the traj labels on the 
            map (instead of placing them in the legend)
    :param  inc_ind - int of which index of path to label if increments is True
    (by default the last point in the trajectory is labeled)
    :param  inc_fontsize - fontsize for trajectory annotations
    :param  text_locs - list of (X,Y) 2-tuple offsets from each of the endpoints 
            for where to place the labels on the map. By default, this is 
            (-12, -12) for each point
    :param  colors - list(str) the color to use for each trajectory
    :param  markers - list(str) the marker to use for each trajectory

    --- Zero / First Point Discrimination ---
    :param  zero_lab - str what to label the zero point if zero_sep is true
    :param  zero_sep - Bool whether to color the zeroth point separately from 
            everything else
    :param  zero_color - str the color of the zeroth point
    :param  zero_incs - list(int) which trajectories to label as 0

    --- Axis Formatting ---
    :param  xlim - 2-tuple of axis min, max
    :param  ylim - 2-tuple of axis min, max
    :param  xrot - int for angle to rotate the x-axis labels
    :param  figsize - 2-tuple for (max) size of canvas

    --- Confidence Regions ---
    {     these only apply if plot_info is NOT None     }
    :param  sd_mult - the number of SDs to use in the confidence intervals
    :param  var_col - 2-tuple of colors for PC1 and PC2 variances
    :param  var_width - float how wide to make the variance bars
    :param  mean_color - the color of the mean plot (if solid). This applies if 
            color_traj is None
    :param  mean_symbol - str symbol to use to mark the mean
    :param  mean_mark_size - int size of mean marker
    :param  mean_lab - str label to give mean trajectory

    --- Figure Saving ---
    :param  save - bool whether to save the image
    :param  savename - str filename for saving
    
    """
    # align the coordinates if they need to be aligned
    if align_coords:
        coordinates = align_traj_to_x(coordinates, yflip=yflip, xflip=xflip, 
                                    zero_ind=zero_ind)

    inc_fontsize = inc_fontsize if inc_fontsize else fontsize

    """If there is plotinfo for variance"""
    if plot_info is not None:
        xs_in = plot_info['xs']
        ys_in = plot_info['ys']

        # parse the dict
        # xs = plot_info['xs']
        # ys = plot_info['ys']
        u1 = plot_info['u1']
        v1 = plot_info['v1']

        u2 = plot_info['u2']
        v2 = plot_info['v2']

        # Lengths of the Arrow
        # smaller values make the arrows longer
        # as per matplotlib quiver documentation: the scale is inverse
        s1 = []
        for k in plot_info['s1']:
            if k != 0: 
                s1.append(1/(k * sd_mult) )
            else: 
                s1.append(np.inf)
        s2 = []
        for k in plot_info['s2']:
            if k != 0: 
                s2.append(1/(k * sd_mult)) 
            else: 
                s2.append(np.inf)
        
        mean_coords = np.array([xs_in, ys_in]).T
        #print(mean_coords)

        # align the coordinates if true and flip as indicated
        if align_coords:
            mean_coords = align_traj_to_x(mean_coords, yflip=yflip, xflip=xflip, 
                                   zero_ind=zero_ind)
            # adjusting the axes as necessary
            if yflip:
                u1 = [-k for k in u1]
                u2 = [-k for k in u2]
                # v1 = [-k for k in v1]
                # v2 = [-k for k in v2]
            if xflip:
                v1 = [-k for k in v1]
                v2 = [-k for k in v2]
                # u1 = [-k for k in u1]
                # u2 = [-k for k in u2]
    
    # set the default color map
    color_map = plt.cm.plasma #if color_map is None else color_map

    if color_traj is not None and color_traj != True: # taking in a color map
        assert steps is not None, "to use color_traj, you must include steps"
        # for i in range(len(steps)):
        #     if np.max(steps[i]) != 1: # normalize if necessary
        #         steps[i] /= np.max(steps[i])
        if cb_norm and 'log' not in cb_norm:
            lims_cb = (np.min([np.min(k) for k in steps]), np.max([np.max(k) for k in steps]))
        else:
            print('Using setting for bar_range since log scale being used')
            lims_cb = bar_range
        # print(lims_cb)
        if accuracies is None:
            bar_range=lims_cb
        
        mean_traj = color_traj
        if traj_a != 1:
            assert traj_a < 1 and traj_a >= 0, "custom cmap alpha must be in [0,1)"
            color_traj = matplotlib.colors.ListedColormap([(r,g,b,traj_a) for 
                                                r, g, b, _ in 
                                                color_traj(np.arange(color_traj.N))])
            print('ALPHA ADAPTED')
    if color_traj == True:
        color_traj = color_map

    # setting the number of points in each trajectory
    if n_models:
        split_indices = [0]+[sum(n_models[:i]) for i in range(1,len(n_models)+1)]
    else: 
        split_indices = [0, len(coordinates) -1]
    
    # number of different trajectories
    n_perturbations = len(n_models) if n_models else 1

    colors = colors if colors is not None else \
             plt.cm.viridis(np.linspace(0, 1, n_perturbations)) 
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

    """Plotting Each Trajectory"""
    for i in range(n_perturbations):
        low = split_indices[i]
        hig = split_indices[i+1]
        xs = coordinates[low:hig, 0]
        ys = coordinates[low:hig, 1]
        #print(increments[i])
        
        if markers is None:
            mark = 'o'
        else:
            if i < len(markers):
                mark = markers[i] if markers[i] else 'o'
            else:
                mark = 'o'
        # plotting the trajectory
        if zero_sep:
            #print(xs, ys)
            plt.scatter(xs[0], ys[0], marker=mark, s=markersize**2, 
                        color=zero_color, zorder=5)
            if i == 0:
                zero_legend = [matplotlib.patches.Patch(facecolor=zero_color,
                                                        edgecolor=zero_color,
                                                        label=zero_lab)]
        if accuracies or color_traj:
            plt.plot(xs, ys, linestyle=':',
                 color='k', label=labels[i], linewidth=1, zorder=1)
            if accuracies:
                scat = plt.scatter(xs, ys, c=accuracies[i], marker=mark, s=markersize**2,
                            cmap=color_map, vmin=bar_range[0], 
                            vmax=bar_range[1], zorder=2, norm=cb_norm)
                increment_color = color_map(accuracies[i][-1])
            if color_traj and color_list is None:
                step_i = np.array(steps[i])
                step_i[step_i <= 0] = bar_range[0]
                #print(steps, steps[i])
                if cb_norm and 'log' in cb_norm:
                    colors[colors <= 0] = bar_range[0]
                #print(colors)
                scat = plt.scatter(xs, ys, c=step_i, cmap=color_traj, 
                            marker=mark, s=markersize**2, norm=cb_norm,
                            vmin=bar_range[0], vmax=bar_range[1], zorder=2)
                increment_color = color_traj(step_i[-1])
            if color_traj and color_list:
                scat = plt.scatter(xs, ys, c=color_list[i], marker=mark, s=markersize**2,
                zorder=3)
                increment_color = color_list[i][-1]
        else:
            plt.plot(xs, ys, markersize=markersize, marker=mark, linestyle=':',
                 color=colors[i], label=labels[i], linewidth=1, zorder=1,
                 mew=0)
            increment_color = colors[i]
        
        # plotting the increments

        if increments and labels:
            # set the ith increments
            increments_i = ['']*(n_models[i])
            increments_i[inc_ind] = f'{labels[i]}'
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
                                bbox=dict(boxstyle='round,pad=0.05', 
                                          fc=increment_color, alpha=0.2),
                                arrowprops=dict(arrowstyle='-', 
                                                connectionstyle='arc3,rad=0'),
                                fontsize=inc_fontsize)
    
    # setting the labels for the trajectories as appropriate
    if labels and not increments:
        if legend_order:
            handles, labels = plt.gca().get_legend_handles_labels()
            ax.legend([handles[idx] for idx in legend_order],
                    [labels[idx] for idx in legend_order], loc='upper left', 
                    bbox_to_anchor=(0.1, -0.08), fontsize=fontsize, ncol=legend_cols)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0.1, -0.08), fontsize=fontsize, 
                ncol=legend_cols)
    if zero_lab:
        ax.legend(handles=zero_legend, fontsize=fontsize)

    """Plot the means"""
    if plot_info is not None:
        #print(mean_coords)
        if color_traj:
            step_i = np.array(steps[0])
            if cb_norm and 'log' in cb_norm:
                colors[colors <= 0] = bar_range[0]
            mean_legend = plt.scatter(mean_coords[:,0], mean_coords[:,1], 
                                      c=step_i, cmap=mean_traj, 
                                      marker=mean_symbol, s=mean_mark_size**2, 
                                      norm=cb_norm, vmin=bar_range[0], 
                                      vmax=bar_range[1], zorder=10000)
            plt.scatter(mean_coords[0,0], mean_coords[0,1], color=zero_color,
                        marker=mean_symbol, s=mean_mark_size**2, zorder=10000)
        else:
            mean_legend = plt.scatter(mean_coords[:,0], mean_coords[:,1], color=mean_color,
                    marker=mean_symbol, s=mean_mark_size**2, label=mean_lab, zorder=10000)
        
        if zero_sep:
        #     plt.scatter(mean_coords[0,0], mean_coords[0,1], marker=mean_symbol, s=100, 
        #                 color=zero_color, zorder=5)
            ax.legend(handles=zero_legend+[mean_legend], fontsize=fontsize, loc=legend_loc)


    """ADDING THE ARROWS"""
    # add in the first axis
    if plot_info is not None:
        for i in range(len(u1)):
            s = s1[i] if s1[i] != 0 else 1
            qv1 = plt.quiver(mean_coords[i, 0], mean_coords[i, 1], u1[i], v1[i], 
                            scale=s, headwidth=1, headlength=0, 
                            scale_units='x', units='dots', width=varwidth, 
                            color=varcols[0],
                            label='', zorder=10000)
            plt.quiver(mean_coords[i, 0], mean_coords[i, 1], -u1[i], -v1[i], 
                        scale=s, headwidth=1, headlength=0,
                        scale_units='x', units='dots', width=varwidth, 
                        color=varcols[0], zorder=10000)
        # second axis
        for i in range(len(u2)):
            s = s2[i] if s2[i] != 0 else 1
            qv2 = plt.quiver(mean_coords[i, 0], mean_coords[i, 1], u2[i], v2[i], 
                            scale=s, headwidth=1, headlength=0,
                            scale_units='x', units='dots', width=varwidth, 
                            color=varcols[1], label='', zorder=10000)
            plt.quiver(mean_coords[i, 0], mean_coords[i, 1], -u2[i], -v2[i], 
                            scale=s, headwidth=1, headlength=0,
                            scale_units='x', units='dots', width=varwidth, 
                            color=varcols[1], zorder=10000)

    """setting the colorbar"""
    if (accuracies or color_traj) and colorbar and not color_list:
        # ticks
        cbar_ticks = steps[0] if color_traj else [np.min(np.array(accuracies)),\
                                                  np.max(np.array(accuracies))]
        # print(steps[0])
        # print(f'FIRST COLORBAR TICK', cbar_ticks[0])
        
        if cbar_ticks[0] == 0:
            cbar_ticks[0] = bar_range[0]
        cbar_ticks = np.array(cbar_ticks).flatten()
        cbar = plt.colorbar(scat, ticks=cbar_ticks)
        cbar.ax.tick_params(labelsize=16)
        
        # define the tick labels
        def tick_format(x):
            # print(x)
            if x == bar_range[0]:
                y = f'0 - init'
            elif int(x) == x:
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
        cbar.set_label(cbar_label, fontsize=fontsize)    
    
    # setting the axis limits if they are given:
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])

    plt.tick_params(axis='both', which='both', labelsize=fontsize)
    plt.tick_params(axis='x', labelrotation=xrot)
    plt.title(title, fontsize=fontsize)
    plt.gca().set_aspect('equal')
    
    if save:
        plt.savefig(savename, bbox_inches='tight')

    plt.show()
    
    return 


##############################################
### color_list and alpha/colorbar creation ###
##############################################

def make_color_list(color_map, color_increments, alpha_increments,
                    alpha_range=(0,1), color_norm=None):
    """
    creates a color_list : list(list(RGBA)) for use in plotting variance plots

    :param color_map: matplotlib.colors.Colormap
    :param color_increments: list(list(float)) - values to get colors  
    :param alpha_increments: list(list(float)) - values to get alphas 
    :param alpha_range: 2-tuple(float) - min and max alpha to range from

    :return rgba_lists : list(list(RGBA)) - rgba values. Same "shape" as 
    color_increments and alpha_increments

    NOTE: color_increments and alpha_increments must be the same "shape"

    """
    # check that color_increments and alpha_increments are compatible
    increment_error = "color_increments and alpha_increments must be the same shape"
    assert len(color_increments) == len(alpha_increments), increment_error
    for i in range(len(color_increments)):
        assert len(color_increments[i]) == len(alpha_increments[i]), increment_error

    # get the min and max values from each of the increment lists
    alpha_min, alpha_max = get_increment_min_max(alpha_increments)
    color_min, color_max = get_increment_min_max(color_increments)

    # make the color norm for the color map
    cnorm = color_norm if color_norm else colors.Normalize(vmin=color_min, vmax=color_max)

    # make the alpha conversion
    get_alpha = make_rescaler(alpha_min, alpha_max, *alpha_range)

    # get the RGB,A color values and put them into a list
    rgba_lists = []
    for i in range(len(color_increments)): # this is a list of color values
        color_list = color_increments[i]
        alpha_list = alpha_increments[i]

        # get the RGB value
        rgb_list = [color_map(cnorm(k)) for k in color_list]
        # add the alpha value
        alpha_vals = [get_alpha(k) for k in alpha_list]
        # get the new colors
        new_color_list = [colors.to_rgba(rgb_list[i], alpha=alpha_vals[i]) for i in range(len(rgb_list))]

        # add to the master list
        rgba_lists.append(new_color_list)
    
    print(color_list_use_message)

    return rgba_lists

color_list_use_message = """!!!IMPORTANT!!! - color_list use information
You are making a color_list for plot_variance_plot. You MUST include additional
arguments into plot_variance_plot at this time based on your inputs to generate 
the list as follows:

color_list = output of this function
color_traj = color_map (the same one you input) or True
steps = color_increments 

(optional):
colorbar = False (default)
    if you keep this true, then the colorbar will NOT match the color_list 
    conventions. To make a colorbar that reflects both the alpha values and the
    colormap, use custom_colorbar.

Yes, it is annoying. Yes, I could fix it. No, I have not. That is a several-
day code optimization project that I am not doing right now. Yes, my code is 
jank. That is why you are seeing this message. (I write, as if anyone else is 
using this spaghetti code)
"""


""" Color List Helper Functions """
def get_increment_min_max(nested_lists):
    min_val = min([min(li) for li in nested_lists])
    max_val = max([max(li) for li in nested_lists])
    return min_val, max_val

def rescale_values(x_old, old_min, old_max, new_min, new_max):
    """ linearly rescales x_old in range (old_min, old_max) to y_new in (new_min, new_max)
    """
    y_new = ((new_max-new_min)/(old_max-old_min))*(x_old-old_max) + new_max
    return y_new

def make_rescaler(old_min, old_max, new_min, new_max):
    """ makes a rescale function with given old range and new range
    f(x_old) = rescale_values(x_old *given_params)
    """
    def rescale_function(x):
        return rescale_values(x, old_min, old_max, new_min, new_max)
    
    return rescale_function

def general_tick_format(x):
    if isinstance(x, str):
        y = x
    elif int(x) == x:
        y = int(x)
    elif 0.01 <= x <= 1e2:
        y = f'{x:.2f}'
    else:
        y = f'{x:.2e}'
    return y


def custom_colorbar(color_map, alphas=False, alpha_range=(0,1), norm=None,
                    color_title='colormap', alpha_title='transparency',
                    ticklist=None, ticklabels=None, alphaticks=None,
                    alphalabels=None, alphanorm=None, orientation="horizontal",
                    fontsize=20, figsize=None, format_labels=True, 
                    alpha_aspect=2, alpha_normalizer=None, xlabel_rotation=0):
    """ Creates a custom colorbar, possibly including alpha values
    :param color_map: matplotlib.pyplot.cm.*name* colormap
    :param alphas: bool - whether to also plot alpha range
    :param alpha_range: tuple(float, float) - range of possible alpha values

    :param norm: matplotlib.colors.Norm(alization) - if the colormap is 
                 normalized somehow
    :param alphanorm: str in ["log", "linear"] - default "log"

    :param color_title: str - the title to go next to the colorbar
    :param alpha_title: str - the title to go next to the transparency bar

    :param ticklist: list - default None, the list of ticks to use for the 
                     colormap. If None, then just (0, 1)
    :param ticklabels: list - default None, the list of labels to use for each
                       tick on the colormap. If None, then the ticklist with
                       default formatting
    
    :param alphaticks: list - same as ticklist but for alpha values
    :param alphalabels: list - same as ticklabels but for alpha values

    :param orientation: str in ["horizontal", "vertical"] - which way the 
                        colormap direction is
    
    :param alpha_aspect: float - proportion of the length of the color bar for
                                 the alpha bar
    
    :param fontsize: int - the fontsize used in the plot
    :param figsize: tuple(float, float) - the size of the final figure
    :param format_labels: bool - whether to format the labels with default 
    
    """
    ### Check input compatibility
    # TODO: Add compatibility checks

    default_sizes = {
        (True, "horizontal"): (8,8), 
        (True, "vertical"): (8,8),
        (False, "horizontal"): (8,1.5),
        (False, "vertical"): (1.5,8)
    }
    # make the figure
    figsize = figsize if figsize else default_sizes[alphas, orientation]
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    norm = norm if norm is not None else colors.Normalize(0, 1)

    # set the normal colorbar/colormap ticks
    ticklist = ticklist if ticklist is not None else (0, 1)
    if format_labels and ticklabels:
        ticklabels=[general_tick_format(k) for k in ticklabels]
    ticklabels = ticklabels if ticklabels else [general_tick_format(k) for \
                                                k in ticklist]
    # make so ticks are in same norm
    ticklist = [norm(k) for k in ticklist]
    # print(ticklist, ticklabels)

    if not alphas:
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=color_map),
                            cax=ax, orientation=orientation, ticks=ticklist)
        cbar.set_label(color_title, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        # set ticks/labels labels
        if orientation == 'horizontal':
            cbar.ax.set_xticklabels(ticklabels)
        else:
            cbar.ax.set_yticklabels(ticklabels)

    else:
        alpha_normalizer = alpha_normalizer if alpha_normalizer is not None \
                                                    else colors.Normalize(0, 1)
        # get alpha norm
        alphanorm = eval_alphanorm(alphanorm)
        n_alpha_vals = int(color_map.N*alpha_aspect)
        # make a plot showing both alphas and colors
        spacing = {"log": np.geomspace, "linear": np.linspace}
        if alpha_range[0] == 0 and alphanorm == "log":
            alpha_range = (1e-4, alpha_range[1])
        alphas = spacing[alphanorm](alpha_range[0], alpha_range[1], 
                                    n_alpha_vals)
        # print(min(alphas), max(alphas))
        
        # get the ticks for alphas
        alphaticks = alphaticks if alphaticks else alpha_range
        if format_labels and alphalabels:
            alphalabels = [general_tick_format(k) for k in alphalabels]
        alphalabels = alphalabels if alphalabels else \
            [general_tick_format(k) for k in alphaticks]
        alphaticks = [alpha_normalizer(k) for k in alphaticks]
        # print(alphaticks, alphalabels)
        
        # Get the RGBA values
        color_array = color_map(np.arange(color_map.N))
        rgbas = []
        for a in alphas:
            alpha_colors = color_array.copy()
            alpha_colors[:,-1] = np.repeat(a, color_map.N)
            rgbas.append(alpha_colors)
        
        rgbas = np.array(rgbas)

        # get the orientation
        transpose_indices = {"horizontal": [0,1,2], "vertical": [1,0,2]}
        rgbas = np.transpose(rgbas, axes=transpose_indices[orientation])

        # show the image
        plt.imshow(rgbas, origin="lower")
        # put the y ticks on the right
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        if orientation == "horizontal":
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

        # do the colormap ticks
        # tick locations need to be changed
        reverse_color = colors.Normalize(0, 256)
        ticklist = [reverse_color.inverse(k) for k in ticklist]
        color_axis = {"horizontal": plt.xticks, "vertical": plt.yticks}
        color_axis[orientation](ticklist, ticklabels)
        # set the colorbar caption
        set_color_title = {"horizontal": plt.xlabel, "vertical": plt.ylabel}
        set_color_title[orientation](color_title, fontsize=fontsize)

        # do the alpha ticks
        # tick locations need to be changed
        rescale_alpha = make_rescaler(alpha_range[0], alpha_range[1], 0, 1)
        reverse_alpha = colors.Normalize(0, n_alpha_vals)
        alphaticks = [reverse_alpha.inverse(k) for k in alphaticks]
        alpha_axis = {"horizontal": plt.yticks, "vertical": plt.xticks}
        alpha_axis[orientation](alphaticks, alphalabels)
        # set the alpha bar caption
        set_alpha_title = {"horizontal": plt.ylabel, "vertical": plt.xlabel}
        set_alpha_title[orientation](alpha_title, fontsize=fontsize)

    # set the tick params
    plt.tick_params(axis="both", which="both", labelsize=fontsize)
    plt.xticks(rotation=xlabel_rotation)
    
    plt.show()
    return


def eval_alphanorm(input):
    if input is None:
        return "log"
    elif input in ["log", "linear"]:
        return input
    else:
        raise ValueError("alphanorm must be in ['log','linear']. Default: 'log'")



class VariancePlotEvo:

    def __init__(self, *args, names=None):
        """
        :param args     :   these should be VariancePlot objects
        """
        self.plot_objs = args

        if names is not None:
            assert len(args) == len(names), \
                'Number of names must equal number of trajectories'
            self.names = names
        else:
            self.names = list(range(1, len(args)+1))

        for plot in self.plot_objs:
            plot.get_variance_plot_info()
            # now each plot has .plot_info not set to None

        return
    
    def plot_variance_evo(self, title=None, tick_locs=None, xlog=False,
                          xlabel='X'):
        s_list1 = []
        s_list2 = []
        for plot in self.plot_objs:
            s_list1.append(np.mean([k**2 for k in plot.plot_info['s1']]))
            s_list2.append(np.mean([k**2 for k in plot.plot_info['s2']]))
        
        if tick_locs is None:
            tick_locs = list(range(len(s_list2)))
        fig = plt.figure(figsize=(10, 5))
        plt.plot(tick_locs, s_list1, label='PC1', 
                marker='o', linewidth=0.5)
        plt.plot(tick_locs, s_list2, label='PC2',
                marker='o', linewidth=0.5)

        if title is None:
            title = 'Variance Evolution'
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel('Average Interval Variance', fontsize=16)
        plt.tick_params(axis='both', which='both', labelsize=16)
        
        
        plt.xticks(tick_locs, self.names)

        if xlog:
            plt.xscale('log')

        plt.yscale('log')
        plt.legend(fontsize=16)

        plt.show()

        return
    
"""Some Helper Functions"""
def split_indices(n_checkpoints, n_trials):
    """
    converts the indices for splitting in a similarity matrix plot
    """
    checks = [n_trials] * n_checkpoints
    splits = []
    for i in range(len(checks)-1):
        splits.append(np.sum(checks[:i+1]))

    return splits


###################################
### Multiple Network Comparison ###
###################################

class MultiNetworkComparison:

    def __init__(self, networks: dict, comparison_name=None, save_filename=None,
                 rel_path=''):
        """
        :param networks:    [string] -> SpectrumAnalysis

        Makes a directory for all calculated quantities. 
        These are tracked by comparison_info.json
        """

        # set the networks
        self.networks = networks # dictionary
        self.keys = list(self.networks.keys())
        self.models = list(self.networks.values())
        self.n_models = len(self.keys)
        
        self.layers = range(1,(self.models[0].n_layers)+1)

        self.loader = None

        self.model_paths = dict(zip(self.keys, [model.save_dir for model in self.models]))

        # set the name of the comparison
        if comparison_name is None:
            name = f'{self.keys[0]}compare_x{len(self.keys)}'
        else:
            name = f'{comparison_name}compare_x{len(self.keys)}'
        self.name = name

        # set the save location
        if save_filename is None:
            save_filename = f'{self.name}_storage'
            # assert list(self.networks.keys()) == self.get_disc_info('keys')
        self.save_filename = save_filename
            
        self.rel_path = rel_path
        # set the save filename
        self.disc_location = f'{self.rel_path}{GLOBAL_COMPARISON_DICT}/{self.save_filename}'
        self.disc_info_file = f'{self.disc_location}/comparison_info.json'

        # these all get set below from the disc info
        self.activation_datasets = None
        self.alignment_datasets = None
        self.activation_similarity_datasets = None

        # disc saving
        if os.path.exists(self.disc_location):
            # self.disc_info = self.load_disc_info()
            print('some quantities already on disc')
            self.load_from_file()
        else:
            os.mkdir(self.disc_location)
            self.reset_disc_info()
    
        # calculated later:
        self.weights = self.get_weights()
        self.weight_covariances = self.get_weight_covariances() 
        self.alignments = None
        return
    

    def reset_disc_info(self):
        """Resets disc info to default"""
        disc_info_json = {
            "name" : self.name,
            "model_paths" : self.model_paths,
            "activation_datasets": [],
            "alignment_datasets": [],
            "activation_similarity_datasets": [],
            "layers" : list(range(1,(self.models[0].n_layers)+1)),
        }

        # disc_info_json.update(dict(zip(none_params, [None]*len(none_params))))
        
        self.disc_info = disc_info_json
        self.save_to_file()
        self.set_attributes_from_disc_info()
        return
    
    def set_attributes_from_disc_info(self):
        for key, value in self.disc_info.items():
            setattr(self, key, value)
        return
    
    def set_disc_info_from_attributes(self):
        for key in self.disc_info.keys():
            self.disc_info[key] = getattr(self, key)
        return

    def update_disc_info(self, update_dict):
        """adds new values to disc info and updates saved version"""
        self.disc_info.update(update_dict)
        self.set_attributes_from_disc_info()
        self.save_to_file()
        return
    
    def load_from_file(self):
        """loads current disc info file into disc_info property"""
        with open(self.disc_info_file, 'r') as file:
            self.disc_info = json.load(file)
        self.set_attributes_from_disc_info()
        return
    
    def save_to_file(self):
        """saves current disc info into file"""
        with open(self.disc_info_file, 'w') as file:
            json.dump(self.disc_info, file)
        self.load_from_file()
        return
    
    def save_attribute_updates(self):
        self.set_disc_info_from_attributes()
        self.save_to_file()
        return

    
    def get_weight_matrices(self, save=True, load=True):
        
        # quantity = "weight_matrices"
        savepath = f'{self.disc_location}/weight_matrices.pkl'

        if load and os.path.exists(savepath):
            with open(savepath, 'rb') as file:
                weight_dict = pickle.load(file)
                weight_dict = decompose_matrix_dict(weight_dict)
        else:
            weight_dict = {}
            for key, net in self.networks.items():
                for layer in range(1, net.n_layers+1):
                    _ = net.get_weights()
                    weights = net.weights[layer-1]
                    # print(weights.shape)
                    weight_dict[key, layer] = DecomposedMatrix(matrix = weights) # weights are always tensors
            # save to disc
            if save:
                with open(savepath, 'wb') as file:
                    dict_to_save = recompose_matrix_dict(weight_dict)
                    pickle.dump(dict_to_save, file)

        return weight_dict
    
    def get_weights(self, layers=None, save=True, load=True):
        weight_dict = self.get_weight_matrices(save=save, load=load)
        if layers == None:
            return_dict = weight_dict
        else:
            return_dict = {}
            for key in self.keys:
                for layer in layers:
                    return_dict[key] = self.weights[key, layer]
        return return_dict
    

    def get_weight_covariances(self, save=True, load=True):
        """calculate the weight covariances"""
        
        savepath = f'{self.disc_location}/weight_covariances.pkl'
        if load and os.path.exists(savepath):
            with open(savepath, 'rb') as file:
                weight_covs = pickle.load(file)
                weight_covs = decompose_matrix_dict(weight_covs)
        else:
            if self.weights is None:
                self.get_weight_matrices()
            weight_covs = {}
            for key, value in self.weights.items():
                weight_covs[key] = decomposed_covariance(value.matrix)


            if save:
                with open(savepath, 'wb') as file:
                    dict_to_save = recompose_matrix_dict(weight_covs)
                    pickle.dump(dict_to_save, file)

        return weight_covs
    

    def set_loader(self, loader_name, dataloader):
        """sets the default loader for the comparison"""
        self.loader = (loader_name, dataloader)
        return
    
    def set_layers(self, layers:list):
        """sets the default layers to use"""
        self.layers = layers
        return

    def get_activation_covs(self, dataloader=None, loader_name=None, layers=None, save=True, 
                            load=True, set_loader=True):
        """
        Get activations for a dataloader for each network
        """
        # quantity="activation_covs"
        if loader_name is None:
            assert self.loader is not None
            print(f'using default loader {self.loader[0]}')
            loader_name = self.loader[0]
            dataloader=self.loader[1]
            set_loader=False

        if layers is None:
            layers = self.layers
        layer_name = get_layer_name(layers)
        savepath = f'{self.disc_location}/activation_covariances_{loader_name}{layer_name}.pkl'

        if load and os.path.exists(savepath):
            with open(savepath, 'rb') as file:
                activation_cov_dict = pickle.load(file)
                activation_cov_dict = decompose_matrix_dict(activation_cov_dict)
        
        else: 
            print(f'computing activation covariances for {loader_name}')
            activation_cov_dict = {}
            for key, model in self.networks.items():
                calculated_cov = model.get_activation_covs(dataloader, layers) # returns a layer, cov dict
                for layer in layers:
                    activation_cov_dict[key, layer] = DecomposedMatrix(matrix=calculated_cov[layer])
            
            if save:
                # save the dict
                with open(savepath, 'wb') as file:
                    dict_to_save = recompose_matrix_dict(activation_cov_dict)
                    pickle.dump(dict_to_save, file)
                # save the loader used
                key = "activation_datasets"
                if loader_name not in self.activation_datasets:
                    self.activation_datasets.append(loader_name)
                self.save_attribute_updates()
        
        if set_loader:
            assert loader_name is not None
            assert dataloader is not None
            self.set_loader(loader_name, dataloader)
                
        return activation_cov_dict
    
    def activation_covariance(self, loader_name=None, dataloader=None, layers=None, 
                              load=True, save=True, set_loader=True):
        """container method /alias for get_activation_covs
        Does not require dataloader input"""
        if loader_name not in self.activation_datasets and loader_name is not None:
            assert dataloader is not None
        
        return self.get_activation_covs(dataloader=dataloader, 
                                        loader_name=loader_name, layers=layers, 
                                        save=save, load=load, set_loader=set_loader)

    def get_alignments(self, save:bool=True, load:bool=True, 
                       dataloader:Optional[torch.utils.data.DataLoader]=None, 
                       layers:Optional[list]=None, 
                       loader_name:Optional[str]=None):
        """
        Calculate alignment matrices between each pair of models
        :param dataloader: the dataloader to use to align the models
        :param loader_name: str - the name of the loader. default None
        :param layers: which layers to align
        :param save: bool - whether to save to disc. default True
        :param load: bool - whether to load if on disc already. default True
        
        """
        quantity='alignments'

        if layers is None:
            layers = self.layers
        layer_name = get_layer_name(layers)

        if loader_name is None and dataloader is None and self.loader is not None:
            loader_name, dataloader = self.loader
        elif loader_name is None and len(self.activation_datasets > 0):
            assert dataloader is not None, "must include a dataloader for this name"
            loader_name = self.activation_datasets[0]
        else:
            raise Exception("please include a loader_name and loader")

        savepath_a = f'{self.disc_location}/alignment_matrices_{loader_name}{layer_name}.pkl'
        savepath_r = f'{self.disc_location}/alignment_r2s_{loader_name}{layer_name}.pkl'

        if os.path.exists(savepath_a) and load:
            with open(savepath_a, 'rb') as file:
                align_dict = pickle.load(file)
            with open(savepath_r, 'rb') as file:
                explain_dict = pickle.load(file)
        else:
            assert dataloader is not None, "alignments not computed for this set, must include a dataloader"
            print(f'computing alignments for {loader_name}')
            align_dict = {} # [key1, key2, layer] -> alignment_matrix
            explain_dict = {} # [key1, key2, layer] -> explained variance from alignment
            #layers_use=[lay-1 for lay in layers]
        
            for i in range(self.n_models):
                for j in range(i+1, self.n_models):
                    key1, key2 = self.keys[i], self.keys[j]

                    # get the alignments
                    alignments, explains = align.compute_alignments_r2s(dataloader, layers, #layers_use, 
                                                    self.networks[key1].model, 
                                                    self.networks[key2].model)
                    
                    for k in range(len(layers)):
                        lay_name = layers[k]
                        align_dict[key1, key2, lay_name] = alignments[k]
                        explain_dict[key1, key2, lay_name] = explains[k]
                        # symmetry
                        align_dict[key2, key1, lay_name] = alignments[k].T
                        explain_dict[key2, key1, lay_name] = explains[k]

            if save:
                # save the dict
                with open(savepath_a, 'wb') as file:
                    pickle.dump(align_dict, file)
                with open(savepath_r, 'wb') as file:
                    pickle.dump(explain_dict, file)
                # save the loader used
                if loader_name not in self.alignment_datasets:
                    self.alignment_datasets.append(loader_name)
                self.save_attribute_updates()

        if save:
            self.alignments = align_dict
            self.r2s = explain_dict

        return align_dict, explain_dict # [key1, key2, layer] -> alignment
    

    def get_activation_eigenvector_similarities(self, layers:Optional[list]=None, 
                                                aligned:bool=True,
                                                # covariances:Optional[dict]=None,
                                                loader_name:Optional[str]=None, 
                                                save:bool=True, load:bool=True,
                                                compare_keys:Optional[list]=None):
        """
        Calculates cosine similarity for the activation eigenvectors. 
        Activation covariances must be saved to disc

        :param compare_keys: list - list of model keys to use in comparison
        """
        if aligned:
            assert self.alignments is not None
        if loader_name is None:
            assert self.loader is not None
            loader_name = self.loader[0]
            _ = self.get_activation_covs(self.loader[1], self.loader[0], 
                                         save=True, set_loader=False)
            _ = self.get_alignments(save=True)
        else:
            assert loader_name in self.activation_datasets
        
        if compare_keys is not None:
            assert set(compare_keys).issubset(self.keys)
        else:
            compare_keys = self.keys

        if layers is None:
            layers = self.layers
        layer_name = get_layer_name(layers) 

        align_name = {True: "aligned", False: "unaligned"}[aligned]
        savepath = f'{self.disc_location}/activation_similarity_{align_name}_{loader_name}{layer_name}.pkl' 

        # check if on disc and load
        if os.path.exists(savepath) and load:
            with open(savepath, 'rb') as file:
                similarity_dict = pickle.load(file)     
        else:
            # get the covariances we want
            covariances = self.activation_covariance(loader_name=loader_name, set_loader=False)

            similarity_dict = {}
            if aligned:
                alignments = self.alignments
            
            # doing the calculation
            for layer in layers:
                for i in range(len(compare_keys)):
                    for j in range(i+1, len(compare_keys)):
                        key1, key2 = compare_keys[i], compare_keys[j]

                        align_mat = alignments[key1, key2, layer] if aligned else None

                        cov1 = covariances[key1, layer]
                        cov2 = covariances[key2, layer]

                        sim_matrix = get_eigenvector_similarities(cov1, cov2, 
                                                        align_matrix=align_mat, 
                                                        aligned=aligned)
                        
                        similarity_dict[key1, key2, layer] = sim_matrix
                        similarity_dict[key2, key1, layer] = sim_matrix.T

            if save:
                with open(savepath, 'wb') as file:
                    pickle.dump(similarity_dict, file)
                if loader_name not in self.activation_similarity_datasets:
                    self.activation_similarity_datasets.append(loader_name)
                self.save_attribute_updates()
        
        return similarity_dict
    

    def get_weight_eigenvector_similarities(self, layers:Optional[list]=None, 
                                            aligned:bool=True,
                                            # covariances:Optional[dict]=None,
                                            save:bool=True, load:bool=True,
                                            compare_keys:Optional[list]=None):
        if aligned:
            assert self.alignments is not None
        
        if compare_keys is not None:
            assert set(compare_keys).issubset(self.keys)
        else:
            compare_keys = self.keys

        if layers is None:
            layers = self.layers
        layer_name = get_layer_name(layers) 

        align_name = {True: "aligned", False: "unaligned"}[aligned]
        savepath = f'{self.disc_location}/weight_similarity_{align_name}_{layer_name}.pkl' 
        # print(savepath)

        # check if on disc and load
        if os.path.exists(savepath) and load:
            with open(savepath, 'rb') as file:
                similarity_dict = pickle.load(file)     
        else:
            # get the covariances we want
            covariances = self.weight_covariances

            similarity_dict = {}
            if aligned:
                alignments = self.alignments
            
            # doing the calculation
            for layer in layers:
                for i in range(len(compare_keys)):
                    for j in range(i+1, len(compare_keys)):
                        key1, key2 = compare_keys[i], compare_keys[j]

                        cov1 = covariances[key1, layer] # DecomposedMatrix
                        cov2 = covariances[key2, layer]

                        # print(key1, key2, layer)

                        # alignment
                        if aligned:
                            if layer == 1:
                                align_mat = torch.eye(cov1.matrix.shape[0])
                            else:
                                align_mat = alignments[key1, key2, layer-1] 
                        else:
                            align_mat = None

                        sim_matrix = get_eigenvector_similarities(cov1, cov2, 
                                                        align_matrix=align_mat, 
                                                        aligned=aligned)
                        
                        similarity_dict[key1, key2, layer] = sim_matrix
                        similarity_dict[key2, key1, layer] = sim_matrix.T

            if save:
                with open(savepath, 'wb') as file:
                    pickle.dump(similarity_dict, file)
                self.save_attribute_updates()
        
        return similarity_dict
        


  
"""
Matrix Things from Florentin
"""

def get_backend(x):
    """ Returns the backend adapted to a given tensor or array. """
    if isinstance(x, torch.Tensor):
        return torch
    else:
        return np


def transpose(matrix):
    """ Transpose a stack of matrices: (*, N, M) to (*, M, N). """
    backend = get_backend(matrix)
    return backend.swapaxes(matrix, -1, -2)


def reconstruct(eigenvalues, eigenvectors, dual_eigenvectors):
    """ eigenvalues is (*, N,), eigenvectors are (*, N, C) and duals are (*, N, D). Returns (*, C, D) matrices.
    Assumes real eigenvalues and eigenvectors. """
    # Reconstruct with eigenvectors.T @ diag(eigenvalues) @ dual_eigenvectors.
    return transpose(eigenvectors) @ (eigenvalues[..., :, None] * dual_eigenvectors)


def decompose(matrix, decomposition="eigh", rank=None):
    """ Performs the decomposition matrix = eigenvectors.T @ diag(eigenvalues) @ dual_eigenvectors.
    matrix is (*, C, D), returns eigenvalues in descending order (*, N), eigenvectors (*, N, C) and their duals (*, N, D).
    N can be smaller than D because we could prune small eigenvalues.
    There are three decompositions:
    - "svd": compute singular values (non-negative) and left and right singular vectors (orthogonal bases)
    - "eig": compute eigenvalues and eigenvectors, dual eigenvectors are the inverse transpose of eigenvectors (requires C = D and real eigenvalues)
    - "eigh": Hermitian case, equivalent to both "svd" and "eig" but faster and more stable (requires C = D)
    rank is an optional upper bound used to prune the number of eigenvalues and eigenvectors.
    """
    def _decompose(matrix):
        backend = get_backend(matrix)
        if decomposition == "svd":
            eigenvectors, eigenvalues, dual_eigenvectors = backend.linalg.svd(matrix, full_matrices=False)
            # Shapes are (*, N), (*, C, N), (*, N, D) with N = min(C, D). Singular values are in descending order.
            eigenvectors = transpose(eigenvectors)  # (*, N, C)
        else:
            sym = dict(eig=False, eigh=True)[decomposition]
            method = backend.linalg.eigh if sym else backend.linalg.eig
            eigenvalues, eigenvectors = method(matrix)  # eigenvalues (*, N) ascending, eigenvectors (*, C, N)

            # Sort in descending order.
            if sym:
                if backend == np:
                    eigenvalues, eigenvectors = eigenvalues[..., ::-1], eigenvectors[..., ::-1]
                else:
                    eigenvalues, eigenvectors = eigenvalues.flip(-1), eigenvectors.flip(-1)
            else:
                assert np.isreal(eigenvalues.dtype)  # Complex eigenvalues not dealt with for now.
                I = backend.argsort(-eigenvalues, axis=-1)  # (*, N)
                take_along = torch.take_along_dim if backend == torch else np.take_along_axis
                eigenvalues, eigenvectors = take_along(eigenvalues, I, axis=-1), \
                                            take_along(eigenvectors, I[..., None, :], axis=-1)

            eigenvectors = transpose(eigenvectors)  # (*, N, C)

            if sym:
                dual_eigenvectors = eigenvectors
            else:
                dual_eigenvectors = transpose(backend.linalg.inv(eigenvectors))  # (*, N, D)

        # Prune eigenvalues that are theoretically zero because of low-rank.
        if rank is not None:
            eigenvalues = eigenvalues[..., :rank]
            eigenvectors = eigenvectors[..., :rank, :]
            dual_eigenvectors = dual_eigenvectors[..., :rank, :]

        # Prune eigenvalues that are too small (deprecated because dual_eigenvectors and batch axes)
        # I = eigenvalues >= eigenvalues[0] / 1e6
        # eigenvalues, eigenvectors = eigenvalues[I], eigenvectors[I]

        return eigenvalues, eigenvectors, dual_eigenvectors
    try:
        eig = _decompose(matrix)
    except RuntimeError as ex:
        print(f"RuntimeError ({ex}) while computing {decomposition} decomposition of shape {matrix.shape}, retrying with numpy")
        matrix_np = matrix.cpu().numpy()
        eigs_np = _decompose(matrix_np)
        eig = tuple(torch.from_numpy(e_np).to(dtype=matrix.dtype, device=matrix.device) for e_np in eigs_np)

    return eig

    
class DecomposedMatrix:
    """
    Class to handle matrix decomposition easily
    """

    def __init__(self, matrix=None, decomposition="eigh", rank=float("inf"),
                 eigenvalues=None, eigenvectors=None, dual_eigenvectors=None):
        
        self._matrix = matrix  # (*, C, D)
        self.decomposition = decomposition
        self.rank: int = min(min(*matrix.shape[-2:]) if matrix is not None else float("inf"), eigenvalues.shape[-1] if eigenvalues is not None else float("inf"), rank)  # Never None.
        self._eigenvalues = eigenvalues  # (*, R), descending
        self._eigenvectors = eigenvectors  # (*, R, C)
        self._dual_eigenvectors = eigenvectors if dual_eigenvectors is None else dual_eigenvectors  # (*, R, D)


        self.backend = get_backend(self._matrix if self._matrix is not None else self._eigenvectors)

        return
    
    def reconstruct(self):
        if self._matrix is None:
            self._matrix = reconstruct(self._eigenvalues, self._eigenvectors, self._dual_eigenvectors)
        return self

    def decompose(self):
        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors, self._dual_eigenvectors = decompose(self._matrix, decomposition=self.decomposition, rank=self.rank)
            # Sets the true rank (min(rank, C, D)) as opposed to the optional upper bound provided.
            self.rank = self._eigenvalues.shape[-1]
        return self

    @property
    def matrix(self):
        return self.reconstruct()._matrix
    
    @property
    def eigenvalues(self):
        return self.decompose()._eigenvalues

    @property
    def eigenvectors(self):
        return self.decompose()._eigenvectors

    @property
    def dual_eigenvectors(self):
        return self.decompose()._dual_eigenvectors
        
    @property
    def T(self):
        """ Returns a transposed view of this DecomposedMatrix (swaps eigenvectors and dual_eigenvectors). """
        return DecomposedMatrix(
            matrix=self._matrix.mT if self._matrix is not None else None, decomposition=self.decomposition, rank=self.rank,
            eigenvalues=self._eigenvalues, eigenvectors=self._dual_eigenvectors, dual_eigenvectors=self._eigenvectors,
        )

    @property
    def left_singular_vectors(self):
        return self.eigenvalues
    
    @property
    def right_singular_vectors(self):
        return self.dual_eigenvectors
    
    @property
    def singular_values(self):
        return self.eigenvalues

"""
Helper Functions for MultiNetworkCompare
"""

def get_eigenvector_similarities(matrix1 : DecomposedMatrix, 
                                 matrix2 : DecomposedMatrix, 
                                 align_matrix=None,
                                 aligned=True):
    """
    Gets the (aligned) eigenvector similarities between the two matrices
    M1 = USV^*
    M2 = WTX^*
    Calculates V A X^*
    """
    if aligned:
        assert align_matrix is not None, "must include alignment matrix"
    else:
        size = matrix1.dual_eigenvectors.shape[-1]
        align_matrix = matrix1.backend.eye(size)
    
    return torch.abs(matrix1.dual_eigenvectors @ align_matrix @ matrix2.dual_eigenvectors.T)


def get_layer_name(layer_list):
    """ Converts layer list (of ints) into string name 
    :param layer_list: list(int) 
    """
    layer_strings = [str(k) for k in layer_list]
    layer_post = "j".join(layer_strings)
    return f'j{layer_post}'

def decompose_matrix_dict(diction):
    """
    [key] -> matrix into [key] -> DecomposedMatrix
    """
    new_dict = {}
    for key, value in diction.items():
        new_dict[key] = DecomposedMatrix(matrix=value)

    return new_dict

def recompose_matrix_dict(diction):
    """
    [key] -> DecomposedMatrix into [key] -> matrix
    """
    new_dict = {}
    for key, value in diction.items():
        new_dict[key] = value.matrix

    return new_dict


def decomposed_covariance(A, rank=float("inf"), full_matrix=False):
    backend = get_backend(A)
    assert backend == torch

    n, d = A.shape[-2:]
    rank = min(rank, d, (float("inf") if full_matrix else n))

    if n < d and not full_matrix:
        A = DecomposedMatrix(A, decomposition="svd", rank=rank)

        return DecomposedMatrix(eigenvalues=A.eigenvalues **2/n,
                                eigenvectors=A.dual_eigenvectors,
                                dual_eigenvectors=A.dual_eigenvectors,
                                decomposition="eigh",
                                rank=rank)
    else:
        return DecomposedMatrix(A.T @ A/n, decomposition="eigh", rank=rank)

