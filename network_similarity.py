import spectral_analysis as spec
import alignment as align

import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch

#from geomloss import SamplesLoss

"""
Network similarity object
"""


class network_comparison:

    def __init__(self, net1: spec.spectrum_analysis, net2: spec.spectrum_analysis, names = None):
        """

        :param net1: spectral_analysis.spectrum_analysis object that has been trained already
        :param net2: spectral_analysis.spectrum_analysis object that has been trained already
        """
        self.models = (net1, net2)
        if names:
            self.names = names
        else:
            self.names = ['Network 1', 'Network 2']

        # print the network informationes
        i = 1
        for net in (net1, net2):
            print(f'Network {i}\n=============================')
            print(f'Architecture\n{net.model}')
            print(f'Trained:\t\t{net.epoch_history[-1]} epochs')
            print(f'Trained on:\t\t{net.train_loader}\n======\n=====\n=====')

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

        align_list, r2s = align.compute_alignments(dataloader, layers, self.models[0].model, self.models[1].model)

        align_dict = dict(zip(layers, align_list))
        r2_dict = dict(zip(layers, r2s))

        # set the attributes
        self.alignments = align_dict.copy()
        self.dataloader = dataloader
        self.layers = layers.copy()
        self.r2s = r2_dict.copy()

        # get the alignment covariances and vectors wrt the dataloader for comparison
        for net in self.models:
            _ = net.get_activation_spectrum()

        # get the eigenvectors for the weights
        # 0 is the 0th model, 1 is the 1st model. The weights are then stored in a dictionary with
        # a key corresponding to each layer of the model
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

            # also get the weight covariances for when we might want the distances
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
                  explained_variance=False):
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
    new_diag1[:len(new_vals1), :len(new_vals1)] = torch.diag(new_vals1)

    new_diag2 = torch.zeros((vecs2.size()[0], vecs2.size()[0]))
    new_diag2[:len(new_vals2), :len(new_vals2)] = torch.diag(new_vals2)

    ## multiply out the matrices
    new_mat1 = vecs1.T @ new_diag1 @ vecs1
    new_mat2 = vecs2.T @ new_diag2 @ vecs2

    ## Get the square roots
    sq_diag1 = torch.sqrt(new_diag1)
    sq_mat1 = vecs1.T @ sq_diag1 @ vecs1

    sq_diag2 = torch.sqrt(new_diag2)
    sq_mat2 = vecs2.T @ sq_diag2 @ vecs2

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
