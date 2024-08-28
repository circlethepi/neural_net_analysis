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
def compute_pairwise_sims(model_set, dataloader=None, layer=1, w_clip=30, a_clip=64, 
                          similarity=True, labels=None, model_set2=None):
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
            simobj = sim.network_comparison(model1, model2) 
                                            # names=(names[i], names[j]))

            # get the alignments
            #set_seed(COMMON_SEED)
            #simobj.compute_alignments(model1.train_loader, [layer])
            simobj.compute_alignments(dataloader, [layer])
            #simobj.compute_cossim()

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
                           figsize=(10,10), save=False, saveloc='../image_hold',
                           split_color='r'):
    """
    plots a similarity matrix heatmap
    """
    fig = plt.figure(figsize=figsize)

    mask = np.triu(np.ones_like(sims, dtype=bool), k=0)
    
    mappy = plt.imshow(np.ma.array(sims, mask=mask), cmap='binary', 
                       vmin=vrange[0], vmax=vrange[1],
                        interpolation='nearest')

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

    plt.show()
    
    if save:
        savepath = f'{saveloc}/MAT{title}'
        plt.savefig(savepath)

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

    # first, convert into a dissimilarity matrix
    dissims = np.ones(similarity_matrix.shape) - similarity_matrix

    # compute the MDS
    mds = manifold.MDS(n_components=2, dissimilarity='precomputed', eps=1e-16,
                       max_iter=1000, n_init=100, random_state=0, normalized_stress=False)
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
                    xlim=None, ylim=None, xrot=0, show=True):
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
        plt.savefig(savepath)
    
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

        print('eigenvalues : ', vals)
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

    def get_variance_plot_info(self):
        plot_info = get_variance_axes(self.coordinates)
        self.plot_info = plot_info
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
                      ticks=None, ticklabs=None, xlab='Step', xrot=0):
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

        plt.show()

        return
    

def plot_variance_plot(coordinates, plot_info=None, title="Variance Plot", sd_mult=2,
                       n_models=None, labels=None, increments=None, 
                       text_locs=None, colors=None, legend_cols=2, 
                       legend_order=None, markers=None, markersize=5,
                       legend_loc='best',

                       accuracies=None, 
                       bar_range=(0,1), color_traj=None, steps=None, 
                       cb_norm=None, 
                       zero_incs=[0], figsize=(12, 10), 
                       zero_sep=False, zero_lab=None, zero_color='red', 

                       align_coords=True, yflip=False, xflip=False, zero_ind=None,

                       save=False, saveloc='../image_hold', 
                       xlim=None, ylim=None, xrot=0, 
                       varwidth=1e-3, 

                       mean_color='k', varcols=('#757575', '#757575'),
                       mean_symbol='D', mean_mark_size=5,
                       mean_lab='Mean Trajectory'
    ):
    """
    :param plot_info - output from get_variance_axes
    """
    # align the coordinates if they need to be aligned
    if align_coords:
        coordinates = align_traj_to_x(coordinates, yflip=yflip, xflip=xflip, 
                                    zero_ind=zero_ind)

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
            plt.plot(xs, ys, markersize=markersize, linestyle=':',
                 color='k', label=labels[i], linewidth=1, zorder=1)
            if accuracies:
                scat = plt.scatter(xs, ys, c=accuracies[i], marker=mark, s=markersize**2,
                            cmap=color_map, vmin=bar_range[0], 
                            vmax=bar_range[1], zorder=2, norm=cb_norm)
                increment_color = color_map(accuracies[i][-1])
            if color_traj:
                step_i = np.array(steps[i])
                #print(steps, steps[i])
                if cb_norm and 'log' in cb_norm:
                    colors[colors <= 0] = 1e10-4
                #print(colors)
                scat = plt.scatter(xs, ys, c=step_i, cmap=color_traj, 
                            marker=mark, s=markersize**2, norm=cb_norm,
                            vmin=bar_range[0], vmax=bar_range[1], zorder=2)
                increment_color = color_traj(step_i[-1])
        else:
            plt.plot(xs, ys, markersize=markersize, marker=mark, linestyle=':',
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
    
    # setting the labels for the trajectories as appropriate
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

    """Plot the means"""
    if plot_info is not None:
        #print(mean_coords)
        mean_legend = plt.scatter(mean_coords[:,0], mean_coords[:,1], color=mean_color,
                    marker=mean_symbol, s=mean_mark_size**2, label=mean_lab, zorder=10000)
        
        if zero_sep:
        #     plt.scatter(mean_coords[0,0], mean_coords[0,1], marker=mean_symbol, s=100, 
        #                 color=zero_color, zorder=5)
            ax.legend(handles=zero_legend+[mean_legend], fontsize=16, loc=legend_loc)


    """ADDING THE ARROWS"""
    # add in the first axis
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

    # setting the colorbar
    if accuracies or color_traj:
        # ticks
        cbar_ticks = steps[0] if color_traj else [np.min(np.array(accuracies)), np.max(np.array(accuracies))]
        cbar = plt.colorbar(scat, ticks=cbar_ticks)
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
        plt.savefig(savepath)
    
    return 


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