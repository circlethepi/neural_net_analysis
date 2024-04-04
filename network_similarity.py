import spectral_analysis as spec
import alignment as align

import numpy as np
import matplotlib.pyplot as plt
import scipy

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
            print(f'Trained on:\t\t{net.train_loader}')

            # getting the weight spectrum if not set
            if not net.weight_spectrum:
                net.get_weight_spectrum()

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
        weight_eig_dict = {0: None, 1: None}
        act_eig_dict = {0: None, 1: None}
        weight_spec_dict = {0: None, 1: None}
        act_spec_dict = {0: None, 1: None}
        i = 0
        for net in self.models:
            for quantity in ['activations', 'weights']:
                # set the covariance list
                if quantity == 'activations':
                    covlist = [net.activation_covs[i] for i in layers]
                    self.activation_covs = dict(zip(self.layers, covlist.copy()))
                else:
                    covlist = [net.weight_covs[i-1] for i in layers]
                    self.weight_covs = dict(zip(self.layers, covlist.copy()))

                # get the eigenvectors
                vectors = []        # eigenvector container
                values = []         # values container
                for cov in covlist:
                    vals, vecs = np.linalg.eigh(cov)
                    vals, vecs = np.flip(vals), np.flip(vecs)
                    vecs = vecs.transpose()

                    vectors.append(vecs)
                    values.append(vals)

                # setting the appropriate quantity
                if quantity == 'activations':
                    act_eig_dict[i] = dict(zip(layers, vectors))
                    act_spec_dict[i] = dict(zip(layers, values))
                else:
                    weight_eig_dict[i] = dict(zip(layers, vectors))
                    weight_spec_dict[i] = dict(zip(layers, vectors))
            i += 1

        # set the attributes for the vectors
        self.weight_eigenvectors = weight_eig_dict.copy()
        self.activation_eigenvectors = act_eig_dict.copy()
        self.weight_spectrum = weight_spec_dict.copy()
        self.activation_spectrum = act_spec_dict.copy()
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
                    a_sim = np.abs(a_eigs1 @ a_align @ a_eigs2.transpose())
                else:
                    a_sim = np.abs(a_eigs1 @ a_eigs2)

                sim_mats[(layer, 'activations', aligned)] = a_sim.copy()

                # get the weights similarity
                if aligned and layer != 1:
                    w_sim = np.abs(w_eigs1 @ w_align @ w_eigs2.transpose())
                else:
                    w_sim = np.abs(w_eigs1 @ w_eigs2)
                sim_mats[(layer, 'weights', aligned)] = w_sim.copy()

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
                    aligned = 'aligned ' if aligned else ''
                    r2 = f'\n{self.r2s[layer]}' if quantity == 'activations' else ''
                    title = f'Cosine similarity of {aligned}{quantity} eigenvectors\nLayer {layer}{r2}'

                    # getting the similarity matrix
                    sim_to_plot = self.cossim[(layer, quantity, aligned)]

                    # making the plot
                    fig = plt.figure(figsize=(8,8))
                    mappy = plt.imshow(sim_to_plot[:clip, :clip], cmap='plasma', vmin=0, vmax=1,
                                       interpolation='nearest')
                    plt.colorbar(mappy, fraction=0.045)
                    plt.ylabel(f'Rank - {self.names[0]}')
                    plt.xlabel(f'Rank - {self.names[1]}')

                    plt.title(title)
                    plt.show()
        return

    def network_distance(self, clip=None):
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
            w_align = self.alignments[layer-1] if layer!= 1 else None
            a_align = self.alignments[layer]

            # calculating the aligned eigenvectors for matrix 2
            ## weights
            w_vecs2_aligned = w_vecs2 @ w_align.transpose() if w_align else w_vecs2.copy()
            a_vecs2_aligned = a_vecs2 @ a_align.transpose()


            # first, if clip, clip the covariance matrices
            if clip:
                # weights
                w_cov1 = align.truncate_mat(clip, w_spec1, w_vecs1)







        return


def bw_dist(mat1, mat2):
    print('mat2 shape = ', np.shape(mat2))
    mat1_12 = np.array(scipy.linalg.sqrtm(mat1))
    mult = mat1_12 @ mat2 @ mat1_12
    mult_12 = np.array(scipy.linalg.sqrtm(mult))
    trace1 = np.trace(mat1)
    trace2 = np.trace(mat2)
    tracemult = 2*(np.trace(mult_12))

    bw_dist = trace1 + trace2 - tracemult
    bw_dist = np.sqrt(bw_dist)

    return bw_dist