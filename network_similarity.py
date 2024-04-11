import spectral_analysis as spec
import alignment as align

import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch

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

    def plot_sims(self, clip=50, layers=None, quantities=('activations', 'weights'), alignments=(False, True)):
        if not layers:
            layers = self.layers

        for layer in layers:
            for quantity in quantities:
                for aligned in alignments:
                    # making the title of the figure
                    align_title = 'aligned ' if aligned else ''
                    r2 = f'\n{100*self.r2s[layer]:.3f}% of Variation explained by alignment' \
                        if quantity == 'activations' and aligned else ''
                    title = f'Cosine similarity of {align_title}{quantity} eigenvectors\nLayer {layer}{r2}'

                    # getting the similarity matrix
                    sim_to_plot = self.cossim[(layer, quantity, aligned)]

                    # making the plot
                    fig = plt.figure(figsize=(8,8))
                    mappy = plt.imshow(sim_to_plot.detach().numpy()[:clip, :clip], cmap='plasma', vmin=0, vmax=1,
                                       interpolation='nearest')
                    plt.colorbar(mappy, fraction=0.045)
                    plt.ylabel(f'Rank - {self.names[0]}')
                    plt.xlabel(f'Rank - {self.names[1]}')

                    plt.title(title)
                    plt.show()
        return

    def network_distance(self, clip=None):
        weights = []
        activations = []
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


            # first, if clip, clip the covariance matrices
            if clip:
                # weights
                w_cov1 = align.truncate_mat(clip, w_spec1, w_vecs1.T)
                w_cov2 = align.truncate_mat(clip, w_spec2, w_vecs2_aligned.T)

                # activations
                a_cov1 = align.truncate_mat(clip, a_spec1, a_vecs1.T)
                a_cov2 = align.truncate_mat(clip, a_spec2, a_vecs2_aligned.T)
            else:
                w_cov1 = w_vecs1.T @ w_spec1 @ w_vecs1
                w_cov2 = w_vecs2_aligned.T @ torch.diag(w_spec2) @ w_vecs2_aligned

                a_cov1 = a_vecs1.T @ a_spec1 @ a_vecs1
                a_cov2 = a_vecs2_aligned.T @ torch.diag(a_spec2) @ a_vecs2_aligned

            # compute the distance
            act_bw = bw_dist(a_cov1.detach().numpy(), a_cov2.detach().numpy())
            way_bw = bw_dist(w_cov1.detach().numpy(), w_cov2.detach().numpy())

            activations.append(act_bw)
            weights.append(way_bw)

        print('BW weights calculated. Returning activations and weights in layer order')
        return activations, weights


def bw_dist(mat1, mat2):
    #print('mat2 shape = ', np.shape(mat2))

    # get sqrt of mat1
    #vals1, vecs1 = torch.linalg.eig(mat1)
    #sqrtvals = torch.sqrt(vals1)
    #mat1_12 = vecs1 @ torch.diag(sqrtvals) @ vecs1.T
    mat1_12 = np.array(scipy.linalg.sqrtm(mat1))
    #print(type(mat1_12), np.shape(mat1_12), np.shape(mat2))
    #print(mat1_12)

    # find the mult matrix
    mult = mat1_12 @ mat2 @ mat1_12
    #print(mult)
    mult_12 = np.array(scipy.linalg.sqrtm(mult))

    # get the square root of the mult matrix
    #vals_m, vecs_m = torch.linalg.eigh(mult)
    #sqrtvals_m = torch.sqrt(vals_m)
    #mult_12 = vecs_m @ torch.diag(sqrtvals_m) @ vecs_m.T

    # traces
    trace1 = np.trace(mat1)
    trace2 = np.trace(mat2)
    tracemult = 2*(np.trace(mult_12))

    bw_dist = trace1 + trace2 - tracemult
    bw_dist = np.sqrt(bw_dist)

    return bw_dist