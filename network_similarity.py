from tkinter import FALSE
import spectral_analysis as spec
import alignment as align

import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import time

from tqdm import tqdm

from utils import *

#from geomloss import SamplesLoss
device = set_torch_device()

"""
Network similarity object
"""


class network_comparison:

    def __init__(self, net1: spec.spectrum_analysis, 
                 net2: spec.spectrum_analysis, names = None):
        """

        :param net1: spectral_analysis.spectrum_analysis object that has been trained already
        :param net2: spectral_analysis.spectrum_analysis object that has been trained already

        OR they can be the weights of a network 
        """
        self.models = (net1, net2)
        if names:
            self.names = names
        else:
            self.names = ['Network 1', 'Network 2']

        # print the network informationes
        i = 1
        for net in (net1, net2):
            # print(f'Network {i}\n=============================')
            # print(f'Architecture\n{net.model}')
            # print(f'Trained:\t\t{net.epoch_history[-1]} epochs')
            # print(f'Trained on:\t\t{net.train_loader}\n======\n=====\n=====')

            # getting the weight spectrum if not set
            if not net.weight_spectrum:
                net.get_weight_spectrum()
            i += 1

        # Attributes set by alignment
        self.activation_covs = None
        self.layers = None
        self.alignments = None
        self.dataloader = None
        self.r2s = None

        # covariances
        self.weight_covs = None
        self.activation_covs = None
        # eigenvectors
        self.weight_eigenvectors = None
        self.activation_eigenvectors = None
        # eigenvalues
        self.weight_spectrum = None
        self.activation_spectrum = None

        #
        self.cossim = None

    def compute_alignments(self, dataloader, layers):
        """
        Computes the alignment matrices for each layer between the models and sets the appropriate attributes for futher comparison. Note that the weight_eigenvectors and activation_eigenvectors are the transposed versions

        :param dataloader: the dataloader wrt whose activations the alignment matrix will be calculated
        :param layers: the layers at which to calculate the alignment matrices
        :return: none
        """
        if 0 in layers:
            raise Exception('0 is not a valid layer index for comparison')
        
        # aligns second model to first
        align_list, r2s = align.compute_alignments(dataloader, layers, 
                                                   self.models[0].model, 
                                                   self.models[1].model)

        align_dict = dict(zip(layers, align_list))
        r2_dict = dict(zip(layers, r2s))

        # set the attributes
        self.alignments = align_dict.copy()
        self.dataloader = dataloader
        self.layers = layers.copy()
        self.r2s = r2_dict.copy()

        # get the alignment covariances and vectors wrt the dataloader 
        # for comparison
        for net in self.models:
            _ = net.get_activation_spectrum()

        # get the eigenvectors for the weights
        # 0 is the 0th model, 1 is the 1st model. The weights are then 
        # stored in a dictionary with a key corresponding to each layer 
        # of the model
        weight_vec_dict = {0: None, 1: None}
        act_vec_dict = {0: None, 1: None}
        weight_spec_dict = {0: None, 1: None}
        act_spec_dict = {0: None, 1: None}
        i = 0
        for net in self.models:
            # first, get the weight eigenvectors
            weight_vectors = []
            for layer in layers:
                weight_list = torch.from_numpy(net.weights[layer - 1])
                u, s, vh = torch.linalg.svd(weight_list, full_matrices=False)
                weight_vectors.append(vh)

            # also get the weight covariances for when we might want 
            # the distances
            weight_covlist = [net.weight_covs[i - 1] for i in layers]
            self.weight_covs = dict(zip(self.layers, weight_covlist.copy()))
            w_spec = []
            for wcov in weight_covlist:
                vals, vecs = torch.linalg.eigh(wcov)
                vals, vecs = vals.flip(-1), vecs.flip(-1)

                w_spec.append(vals)

            # get the activations eigenvectors for each model
            cov_list = [net.activation_covs[i-1] for i in layers]
            vectors = []
            values = []
            for cov in cov_list:
                vals, vecs = torch.linalg.eigh(cov)
                vals, vecs = vals.flip(-1), vecs.flip(-1)
                vecs = vecs.T

                vectors.append(vecs)
                values.append(vals)

            # setting the appropriate quantities
            act_vec_dict[i] = dict(zip(layers, vectors.copy()))
            act_spec_dict[i] = dict(zip(layers, values.copy()))

            weight_vec_dict[i] = dict(zip(layers, weight_vectors.copy()))
            weight_spec_dict[i] = dict(zip(layers, w_spec.copy()))

            i += 1

        self.weight_eigenvectors = weight_vec_dict.copy()
        self.activation_eigenvectors = act_vec_dict.copy()
        self.activation_spectrum = act_spec_dict.copy()
        self.weight_spectrum = weight_spec_dict.copy()
        return

    def compute_cossim(self):
        """
        computes the cosine similarity for each layer for the weights and the activations, both aligned and unaligned
        and sets the appropriate attributes on the object
        :return:
        """
        # check that the alignments have been computed
        if not self.alignments:
            raise Exception('No alignment matrices have been calculated. Run network_comparison.compute_alignments with'
                            ' the desired dataloader and layer list first.')

        sim_mats = {}

        for layer in self.layers:
            a_eigs1 = self.activation_eigenvectors[0][layer]
            a_eigs2 = self.activation_eigenvectors[1][layer]

            w_eigs1 = self.weight_eigenvectors[0][layer]
            w_eigs2 = self.weight_eigenvectors[1][layer]

            a_align = self.alignments[layer]
            if layer != 1:
                w_align = self.alignments[layer-1]

            for aligned in [False, True]:
                # get the activation similarity
                if aligned:
                    #print(type(a_eigs1), type(a_align), type(a_eigs2))
                    a_sim = torch.abs(a_eigs1 @ a_align @ a_eigs2.T)
                else:
                    a_sim = torch.abs(a_eigs1 @ a_eigs2)

                sim_mats[(layer, 'activations', aligned)] = torch.clone(a_sim)

                # get the weights similarity
                if aligned and layer != 1:
                    w_sim = torch.abs(w_eigs1 @ w_align @ w_eigs2.T)
                else:
                    w_sim = torch.abs(w_eigs1 @ w_eigs2.T)
                sim_mats[(layer, 'weights', aligned)] = torch.clone(w_sim)

        # set the attripbute
        self.cossim = sim_mats.copy()
        return

    def plot_sims(self, clips=None,
                  layers=None,
                  quantities=('activations', 'weights'),
                  alignments=[True],
                  plot_clip=None,
                  filename_append=None,
                  ed_plot=False,
                  explained_variance=False, save=False):
        if not layers:
            layers = self.layers
        if not clips:
            clips = [50]*len(layers)

        if len(clips) != len(layers):
            print(len(clips), len(layers))
            raise Exception('number of clips needs to be the same as number of layers to plot')

        i=0
        for layer in layers:
            clip = clips[i]
            for quantity in quantities:
                for aligned in alignments:
                    # making the title of the figure
                    align_title = 'aligned ' if aligned else ''
                    r2 = f'\n{100*self.r2s[layer]:.3f}% of Variation explained by alignment' \
                        if (quantity == 'activations' and aligned and explained_variance) else ''
                    title = f'{align_title}{quantity} eigenvectors\nLayer {layer}\n{r2}'

                    # getting the similarity matrix
                    sim_to_plot = self.cossim[(layer, quantity, aligned)]

                    # making the plot
                    fig = plt.figure(figsize=(8,8))
                    mappy = plt.imshow(sim_to_plot.detach().numpy()[:clip, :clip], cmap='binary', vmin=0, vmax=1,
                                       interpolation='nearest')
                    plt.colorbar(mappy, fraction=0.045)
                    plt.ylabel(f'Rank - {self.names[0]}', fontsize=16)
                    plt.xlabel(f'Rank - {self.names[1]}', fontsize=16)

                    # if weights, show where we clipped for the distances
                    if quantity == 'weights' and ed_plot:
                        dim1 = self.models[0].effective_dimensions[layer - 1][-1]
                        dim2 = self.models[1].effective_dimensions[layer - 1][-1]
                        clip_val = int(np.ceil(min(dim1, dim2)))

                        xs1 = [0, clip-1]
                        ys1 = [clip_val, clip_val]
                        xs2 = [clip_val, clip_val]
                        ys2 = [0, clip-1]

                        plt.plot(xs1, ys1, color='r', linestyle=':', label='EDs')
                        plt.plot(xs2, ys2, color='r', linestyle=':')

                    if plot_clip:
                        xs1 = [0, clip-1]
                        ys1 = [plot_clip, plot_clip]
                        xs2 = ys1
                        ys2 = xs1

                        plt.plot(xs1, ys1, color='b', label='dist clip')
                        plt.plot(xs2, ys2, color='b')

                    if plot_clip or quantity == 'weights':
                        plt.legend()

                    plt.title(title, fontsize=20)

                    # saving the figure
                    if save:
                        path = '/Users/mnzk/Documents/40-49. Research/42. Nets/42.97. Library/image_hold/'
                        filename = (f'{self.names}_{quantity}_{aligned}_{layer}'
                                    f'{"_"+filename_append if filename_append else ""}')

                        plt.savefig(path + filename)
                    plt.show()

        return

    def network_distance(self, w_clip=None, a_clip=None, sim=False, return_quantities=False):
        """

        :param w_clip: the rank at which to clip the weight covariances. Default: None (full rank for each)
        :param a_clip: the rank at which to clip the activation convariances. Default: None (full rank for each)
        :param sim: defail
        :param return_quantities: whether or not to return the trace and norm values from the calculation of the
                                  distances and similarities
        :return: list of distances (or similarities) between the two networks. The ith entry of the list corresponds to
                 the (i+1)th layer of the networks.
        """
        weights = []
        activations = []

        quantities = {'activations': [], 'weights': []}

        # for each layer
        for layer in self.layers:
            # getting the vectors and values for each cov matrix
            ## for the weights
            w_vecs1 = self.weight_eigenvectors[0][layer]
            w_spec1 = self.weight_spectrum[0][layer]

            w_vecs2 = self.weight_eigenvectors[1][layer]
            w_spec2 = self.weight_spectrum[1][layer]

            ## for the activations
            a_vecs1 = self.activation_eigenvectors[0][layer]
            a_spec1 = self.activation_spectrum[0][layer]

            a_vecs2 = self.activation_eigenvectors[1][layer]
            a_spec2 = self.activation_spectrum[1][layer]

            # getting the alignment matrices
            w_align = self.alignments[layer-1] if layer != 1 else None
            a_align = self.alignments[layer]

            # calculating the aligned eigenvectors for matrix 2
            ## weights
            w_vecs2_aligned = w_vecs2 @ w_align.T if w_align is not None else torch.clone(w_vecs2)
            a_vecs2_aligned = a_vecs2 @ a_align.T

            if sim:
                q = 'sim'
            else:
                q = 'dist'

            act_bw = bw_dist_covs(a_vecs1, a_spec1, a_vecs2_aligned, a_spec2,
                                  truncate=a_clip, quant=q, return_quantities=return_quantities)
            way_bw = bw_dist_covs(w_vecs1, w_spec1, w_vecs2_aligned, w_spec2,
                                  truncate=w_clip, quant=q, return_quantities=return_quantities)

            if return_quantities:
                activations.append(act_bw[0])
                weights.append(way_bw[0])
                quantities['activations'].append(act_bw[1])
                quantities['weights'].append(way_bw[1])
            else:
                activations.append(act_bw)
                weights.append(way_bw)

        print('BW weights calculated. Returning activations and weights in layer order')
        if return_quantities:
            return activations, weights, quantities
        else:
            return activations, weights


def bw_dist(mat1, mat2):
    """

    :param mat1:
    :param mat2:
    :return:
    """

    # this runs into error.
    mat1_12 = np.array(scipy.linalg.sqrtm(mat1))
    mat2_12 = np.array(scipy.linalg.sqrtm(mat2))

    # since i am using this for symmetric matrices, I can do:
    #vals1, vecs1 = np.linalg.eigh(mat1)
    #vals1, vecs1 = np.flip(vals1), np.flip(vecs1)
    #mat1_12 = vecs1 @ np.sqrt(vals1) @ vecs1.T


    # from the POT implementation (python optimal transport)
    #output = np.trace(mat1 + mat2 - 2 * np.array(scipy.linalg.sqrtm(np.dot(mat1_12, mat2, mat1_12))))

    output = np.trace(mat1) + np.trace(mat2) - 2*np.linalg.norm(mat1_12 @ mat2_12, ord='nuc')

    #loss = SamplesLoss(loss='sinkhorn', p=2, blur=0.05)
    #output = loss(mat1, mat2)
    return output


def bw_dist_covs(vecs1, vals1, vecs2, vals2, truncate=None, quant='dist', return_quantities=False):
    """
    Calculates the Bures-Wasserstein distance between two covariance matrices C1, C2, given as:
    Tr(C1) + Tr(C2) - 2 ||C1^(1/2) C2^(1/2)||
    Where ||.|| is the nuclear norm (the sum of the singular values)

    :param vecs1: torch.tensor  the eigenvectors of the first covariance matrix
    :param vals1: torch.tensor  the eigenvalues of the first covariance matrix
    :param vecs2: torch.tensor  the eigenvectors of the second covariance matrix
    :param vals2: torch.tensor  the eigenvalues of the second covariance matrix
    :param truncate: int        the rank at which to truncate each of the matrices. If None, then the matrix is not
                                truncated and the distance is calculated for the entire matrix
    :param return_quantities: bool  whether to return the trace and norm quantities from caluclating the distance

    :return: distance: float    the BW distance between the two matrices
    :return: (tr1, tr2, nnorm): (float, float, float)   the quantities used to calculate the distance
    """
    if truncate:
        # get the new values for the truncated matrix
        ## matrix 1
        high1 = vals1[truncate - 1]
        new_vals1 = torch.where(vals1 < high1, 0, vals1)
        ## matrix 2
        high2 = vals2[truncate - 1]
        new_vals2 = torch.where(vals2 < high2, 0, vals2)
    else:
        new_vals1 = vals1
        new_vals2 = vals2

    # calculate the full matrices
    ## get the diagonal matrices
    new_diag1 = torch.zeros((vecs1.size()[0], vecs1.size()[0]))
    new_diag1[:len(new_vals1), :len(new_vals1)] = torch.diag(new_vals1).to(device)

    new_diag2 = torch.zeros((vecs2.size()[0], vecs2.size()[0]))
    new_diag2[:len(new_vals2), :len(new_vals2)] = torch.diag(new_vals2).to(device)

    ## multiply out the matrices
    new_mat1 = vecs1.T.to(device) @ new_diag1.to(device) @ vecs1.to(device)
    new_mat2 = vecs2.T.to(device) @ new_diag2.to(device) @ vecs2.to(device)

    ## Get the square roots
    sq_diag1 = torch.sqrt(new_diag1)
    sq_mat1 = vecs1.T.to(device) @ sq_diag1.to(device) @ vecs1.to(device)

    sq_diag2 = torch.sqrt(new_diag2)
    sq_mat2 = vecs2.T.to(device) @ sq_diag2.to(device) @ vecs2.to(device)

    ## get the quantities
    tr1 = torch.trace(new_mat1).item()
    tr2 = torch.trace(new_mat2).item()
    nnorm = torch.trace(sq_mat1.T @ sq_mat2).item()

    ## calculate the distance with all the pieces
    if quant == 'dist':
        distance = tr1 + tr2 - 2*nnorm
    elif quant == 'sim':
        distance = nnorm / np.sqrt(tr1 * tr2)
    else:
        raise Exception('Please select either "dist" or "sim" for parameter quant')

    if return_quantities:
        return distance, (tr1, tr2, nnorm)
    else:
        return distance


class NetworkComparisonCollection:
    """
    A class for comparing multiple networks/experiments
    Weights and Activations
    """

    def __init__(self, *args, layers=[1], dataloader=None, align_to=None, 
                names=None):
        """
        args is a list of networks to compare. these should be spec.spectrum_analysis objects
        align_to is a spec.spectrum_analysis
        align bool whether to align the networks
        """

        self.models = args
        self.reference = align_to if align_to else args[0]
        dataloader = dataloader if dataloader else self.reference.train_loader

        if names:
            self.names = names
        else: 
            self.names = [f'Network {k}' for k in range(1, len(args))]
        
        # getting the weight spectra if not set
        for mod in self.models:
            if not mod.weight_spectrum:
                mod.get_weight_spectrum()
        if not self.reference.weight_spectrum:
            self.reference.get_weight_spectrum()
        
        w_vecs, w_vals, a_vecs, a_vals = \
            compute_eigenvector_spectrum_list(self.models, dataloader, layers)
        
        # attributes set by alignment
        self.activation_covs = None # dict []
        self.layers = layers # list(int)
        self.alignments = None # dict
        self.dataloader = dataloader # torch.dataloader()
        self.r2s = None # dict

        # covariances
        self.weight_covs = None
        self.activation_covs = None

        # eigenvectors
        self.weight_eigenvectors = w_vecs
        self.activation_eigenvectors = a_vecs

        # eigenvalues
        self.weight_spectrum = w_vals
        self.activation_spectrum = a_vals

        # aligned eigenvectors
        self.weight_aligned_vectors = None
        self.activation_aligned_vectors = None

        self.cossim = None

        self.weight_metric_mat = None
        self.activation_metric_mat = None

        return
    
    def compute_alignments(self, dataloader=None):
        """
        Gets the alignments, r2 values, eigenvectors/spectra for each of the 
        models and set the appropriate features
        
        """
        dataloader = dataloader if dataloader else self.dataloader

        # first, compute alignments
        align_dict, r2_dict = compute_alignment_list(self.reference, self.models,
                                                     dataloader, self.layers)
            # [model index] -> [layer] -> alignment matrix (or r2 value)
        self.alignments = align_dict.copy()
        self.r2s = r2_dict.copy()

        w_vecs = self.weight_eigenvectors
        w_vals = self.weight_spectrum
        a_vecs = self.activation_eigenvectors
        a_vals = self.activation_spectrum
        # set the aligned vectors for each quanitity
        # container for each model's aligned vectors
        aligned_vectors_w = []
        aligned_vectors_a = []

        for i in tqdm(range(len(self.models)), desc='Calculating aligned eigenvectors'):
            # get the collection of eigenvectors for each layer
            w_vecs = self.weight_eigenvectors[i]
            a_vecs = self.activation_eigenvectors[i]

            aligns = align_dict[i]

            # containers for the aligned vectors for the model
            model_layers_aligned_acts = []
            model_layers_aligned_ways = []

            # calculate the alignment
            for lay in layers:
                # get the eigenvectors for the layer
                w_lay_vecs = w_vecs[lay]
                a_lay_vecs = a_vecs[lay]

                # get the alignment matrix
                align_mat_a = aligns[lay]
                align_mat_w = aligns[lay-1] if lay != 1 else None

                # align the vectors
                if align_mat_w is not None:
                    w_lay_vecs = w_lay_vecs @ align_mat_w.T
                
                a_lay_vecs = a_lay_vecs @ align_mat_a.T

                model_layers_aligned_acts.append(a_lay_vecs)
                model_layers_aligned_ways.append(w_lay_vecs)
            
            aligned_vectors_w.append(dict(zip(layers, model_layers_aligned_ways)))
            aligned_vectors_a.append(dict(zip(layers, model_layers_aligned_acts)))

        
        model_inds = list(range(len(self.models)))
        aligned_eigen_weights = dict(zip(model_inds, aligned_vectors_w))
        aligned_eigen_activations = dict(zip(model_inds, aligned_vectors_a))
        # dict:
        # [model ind] -> [layer] -> aligned vectors

        # set the features
        self.weight_aligned_vectors = aligned_eigen_weights
        self.activation_aligned_vectors = aligned_eigen_activations
        
        return

    def compute_aligned_vectors(self, dataloader=None):
        dataloader = dataloader if dataloader else self.dataloader

        aligned_weights, aligned_activations = \
            compute_aligned_vectors(self.reference, self.models, dataloader,
                                    self.layers, self.weight_eigenvectors,
                                    self.activation_eigenvectors)
        self.weight_aligned_vectors = aligned_weights
        self.activation_aligned_vectors = aligned_activations
        
        return

    def compute_cossims_vecs(self):
        """
        compute the cosine similarities between the reference and the models 
        for the aligned and unaligned activations and weights
        """
        error_message = "No alignment matrices have been calculated. Run \
        .compute_alignments with the desired dataloader and layer list first"
        assert self.alignments is not None, error_message

        # fill this in later
        
        return

    def get_network_distance_matrix(self, w_clip=None, a_clip=None, sim=False):
        """
        Gives back the metric matrices as a dictionary:
        [layer] -> similarity matrix
        for each of activations and weights (returned in that order)
        """
        weights = []
        activations = []

        #quants = {'activations' : [], 'weights' : []}

        quantity = 'sim' if sim else 'dist'

        for layer in self.layers:
            layer_sims_acts = []
            layer_sims_ways = []

            for ind in range(len(self.models)):
                i_acts = []
                i_ways = []
                for jnd in range(ind+1, len(self.models)):
                    #print(f'Layer: {layer}\nind:  {ind}\njnd:  {jnd}')
                    # activation info
                    act_vecs1 = self.activation_aligned_vectors[ind][layer]
                    act_vals1 = self.activation_spectrum[ind][layer]

                    act_vecs2 = self.activation_aligned_vectors[jnd][layer]
                    act_vals2 = self.activation_spectrum[jnd][layer]

                    # weight info
                    way_vecs1 = self.weight_aligned_vectors[ind][layer]
                    way_vals1 = self.weight_spectrum[ind][layer]

                    way_vecs2 = self.weight_aligned_vectors[jnd][layer]
                    way_vals2 = self.weight_spectrum[jnd][layer]

                    act_bw = bw_dist_covs(act_vecs1, act_vals1, 
                                          act_vecs2, act_vals2,
                                          truncate=a_clip, quant=quantity,
                                          return_quantities=False)
                    
                    way_bw = bw_dist_covs(way_vecs1, way_vals1, 
                                          way_vecs2, way_vals2,
                                          truncate=w_clip, quant=quantity,
                                          return_quantities=False)


                    #sims for the model
                    i_acts.append(act_bw)
                    i_ways.append(way_bw)
                # sims for the layer
                layer_sims_acts.append(i_acts)
                layer_sims_ways.append(i_ways)
            
            # sims for the layer add to all
            #print(layer_sims_acts)
            weights.append(similarity_matrix_from_lists(layer_sims_ways))
            activations.append(similarity_matrix_from_lists(layer_sims_acts))
        
        weight_matrix_dict = dict(zip(self.layers, weights))
        activation_matrix_dict = dict(zip(self.layers, activations))

        # set the appropriate features
        self.weight_metric_mat = weight_matrix_dict
        self.activation_metric_mat = activation_matrix_dict

        return weight_matrix_dict, activation_matrix_dict

def compute_alignment_list(reference, models_to_align, dataloader, layers, 
                           batch_size=9):
    """
    Compute alignments for multiple models wrt a given reference

    reference : spec.spectrum_analysis
    models_to_align : list(spec.spectrum_analysis)
    dataloader : torch.dataloader
    layers : list(int)

    returns dict of alignments for the given layers for the models to the ref
    indexes the models with integers in order as keys

    return is 
    dict: [model index] -> [layer] -> alignment matrix (or r2 value)
    (reference model has index 0)
    
    """
    # check that layer selection is valid
    assert 0 not in layers, '0 is not a valid layer index for comparison'

    # align_lists = []
    # r2_list = []
    # for model in tqdm(models_to_align, desc='Aligning Models'):
    #     align_li, r2 = align.compute_alignments(dataloader, layers, reference.model, 
    #                                          model.model)
        
    #     model_align_dict = dict(zip(layers, align_li))
    #     model_r2_dict = dict(zip(layers, r2))
        
    #     align_lists.append(model_align_dict)
    #     r2_list.append(model_r2_dict)

    #     del model, model_align_dict, model_r2_dict
    #     clear_memory()

    # # create dictionary to return
    # model_inds = list(range(len(models_to_align)))
    # align_dict = dict(zip(model_inds, align_lists))
    # r2_dict = dict(zip(model_inds, r2_list))

    # return align_dict, r2_dict
    align_dict = {}
    r2_dict = {}
    
    num_models = len(models_to_align)
    
    for i in tqdm(range(0, num_models, batch_size), desc='Aligning Models'):
        end_ind = min(i+batch_size, num_models)
        batch_models = models_to_align[i:end_ind]
        align_lists = []
        r2_list = []

        for model in batch_models:
            align_li, r2 = align.compute_alignments(dataloader, layers, reference.model, model.model)
            
            align_li = [k.to('cpu') for k in align_li]
            model_align_dict = dict(zip(layers, align_li))
            model_r2_dict = dict(zip(layers, r2))
            
            align_lists.append(model_align_dict)
            r2_list.append(model_r2_dict)

            del model, model_align_dict, model_r2_dict
            clear_memory()

        batch_indices = list(range(i, min(i + batch_size, num_models)))
        align_dict.update(dict(zip(batch_indices, align_lists)))
        r2_dict.update(dict(zip(batch_indices, r2_list)))
        
        # Clear memory after each batch
        clear_memory()

    return align_dict, r2_dict

        
def compute_eigenvector_spectrum_list(models_list, dataloader, layers):
    """
    models_list :   list(spec.spectrum_analysis)    the list of models to do 
                                                    the calculation for
    
    returns activation and weights eigenvectors and spectra as 
    dict: [model index] -> [layer] -> eigen vectors (or values)

    model indices are set as the index of the model in the input list
    """

    w_vec_dicts = []
    w_val_dicts = []
    a_vec_dicts = []
    a_val_dicts = []

    for model in tqdm(models_list, desc="Getting Eigenvectors"):
        # make sure weights and activations are there
        _ = model.get_activation_covs(dataloader, layers)
        _ = model.get_activation_spectrum()

        if not model.weight_spectrum:
            model.get_weight_spectrum()

        # do for the weights
        weight_covlist = [model.weight_covs[i-1] for i in layers]
        w_vecs = []
        w_vals = []
        for wcov in weight_covlist:
            vals, vecs = torch.linalg.eigh(wcov)
            vals, vecs = vals.flip(-1), vecs.flip(-1)
            w_vals.append(vals)
            w_vecs.append(vecs)
        # setting the appropriate dictionaries
        w_vec_dicts.append(dict(zip(layers, w_vecs.copy())))
        w_val_dicts.append(dict(zip(layers, w_vals.copy())))
        
        
        # do for the activations
        activation_covlist = [model.activation_covs[i-1] for i in layers]
        a_vecs = []
        a_vals = []
        for acov in activation_covlist:
            vals, vecs = torch.linalg.eigh(acov)
            vals, vecs = vals.flip(-1), vecs.flip(-1)
            a_vals.append(vals)
            a_vecs.append(vecs)
        # setting the appropriate dictionaries
        a_vec_dicts.append(dict(zip(layers, a_vecs.copy())))
        a_val_dicts.append(dict(zip(layers, a_vals.copy())))
        
    model_inds = list(range(len(models_list)))

    w_vec_return = dict(zip(model_inds, w_vec_dicts))
    w_val_return = dict(zip(model_inds, w_val_dicts))
    a_vec_return = dict(zip(model_inds, a_vec_dicts))
    a_val_return = dict(zip(model_inds, a_val_dicts))
    
    return w_vec_return, w_val_return, a_vec_return, a_val_return


def compute_metric_2_networks(vecs1, vals1, vecs2, vals2, align=None, 
                                quantity='dist', truncate=None, return_q=False):
    """
    quantity is 'sim' or 'dist'
    vecs and vals for each of 2 matrices ; alignment matrix to align matrix 2 
    to matrix 1
    """
    if align is not None:
        vecs2 = vecs2 @ align.T 

    metric = bw_dist_covs(vecs1, vals1, vecs2, vals2, truncate=truncate, 
                            quant=quantity, return_quantities=return_q)

    return metric


def similarity_matrix_from_lists(lists):
    # also in perturbation_to_map, but this would cause a circular dependency :)
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


def compute_aligned_vectors(reference, models_to_align, dataloader, layers, 
                           model_weight_eigenvectors, model_activation_eigenvectors,
                           batch_size=9):
    """
    Compute alignments for multiple models wrt a given reference

    reference : spec.spectrum_analysis
    models_to_align : list(spec.spectrum_analysis)
    dataloader : torch.dataloader
    layers : list(int)

    returns dict of alignments for the given layers for the models to the ref
    indexes the models with integers in order as keys

    return is 
    dict: [model index] -> [layer] -> alignment matrix (or r2 value)
    (reference model has index 0)
    
    """
    # check that layer selection is valid
    assert 0 not in layers, '0 is not a valid layer index for comparison'
    #align_dict = {}
    #r2_dict = {}

    aligned_weights_dict = {}
    aligned_activations_dict = {}
    
    num_models = len(models_to_align)
    
    for i in tqdm(range(0, num_models, batch_size), desc='Aligning Models'):
        end_ind = min(i+batch_size, num_models)
        batch_models = models_to_align[i:end_ind]
        #align_lists = []
        align_way_list = []
        align_act_list = []
       # r2_list = []

        j = 0 # index of model in batch
        for model in batch_models:
            align_li, _ = align.compute_alignments(dataloader, layers, 
                                                    reference.model, model.model)
            
            # align the matrices
            k = 0 # layer index
            aligned_act_vecs = []
            aligned_way_vecs = []
            for lay in layer:
                w_vecs = model_weight_eigenvectors[i+j][lay]
                a_vecs = model_activation_eigenvectors[i+j][lay]

                # get the alignment matrices
                align_w = align_li[k-1] if k != 0 else None
                align_a = align_li[k]

                # doing the alignments
                if align_w is not None:
                    w_vecs = w_vecs @ align_w.T
                a_vecs = a_vecs @ align_a.T

                aligned_act_vecs.append(a_vecs)
                aligned_way_vecs.append(w_vecs)

                k += 1
            # make into dicts
            aligned_activations = dict(zip(layers, aligned_act_vecs))
            aligned_weights = dict(zip(layers, aligned_way_vecs))
            #model_align_dict = dict(zip(layers, align_li))
            #model_r2_dict = dict(zip(layers, r2))
            
            aligned_way_list.append(aligned_weights)
            aligned_act_list.append(aligned_activations)
            #align_lists.append(model_align_dict)
            #r2_list.append(model_r2_dict)

            del model, aligned_activations, aligned_weights#,\model_align_dict, model_r2_dict
            clear_memory()

            j += 1

        batch_indices = list(range(i, min(i + batch_size, num_models)))
        #align_dict.update(dict(zip(batch_indices, align_lists)))
        #r2_dict.update(dict(zip(batch_indices, r2_list)))
        aligned_weights_dict.update(dict(zip(batch_indices, aligned_way_list)))
        aligned_activations_dict.update(dict(zip(batch_indices, aligned_act_list)))
        
        # Clear memory after each batch
        clear_memory()

    return aligned_weights_dict, aligned_activations_dict #align_dict, r2_dict
