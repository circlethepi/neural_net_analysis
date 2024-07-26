# the spectral analysis object

# torch things
import torch
from torch import nn
import torch.utils.data.dataset
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets

from tqdm import tqdm
from tqdm.notebook import tqdm

# basics
import numpy as np
from matplotlib import pyplot as plt
import os

# other packages/files
import neural_network as nn_mod
import train_network
import effective_dimensions as eff_dim
import plotting_networks as plotter
import alignment as align
import archive.class_splitter as cs

# check if there is a GPU available
from utils import *
device = set_torch_device()


class spectrum_analysis:
    """
    Holds, saves, trains linear Neural Network for analysis
    """

    def __init__(self, n_neurons, vary=None, n_class=10, input_size=32*32*3, 
                 seed=1234, save=False, exp_name=None, 
                 load=False, path=None, epoch=None,
                 rel_path='../'):
        """
        Initializes model and associated quantitties

        If loading a model, you must still initialize with the number of 
        neurons in each layer desired
        """
        # create the associated model
        arch = [f'fc{k}' for k in n_neurons]
        # set the seed
        if seed is not None:
            set_seed(seed)
        if not load:
            self.model = nn_mod.Neural_Network(n_neurons, num_classes=n_class, 
                                            input_size=input_size)
            # set the experiment name and path
            self.save_dir = f'{rel_path}/model_library/{exp_name}-{"".join(arch)}'
            if save:
                os.mkdir(self.save_dir)
        else:
            assert path is not None and epoch is not None, \
                "Invalid model loading parameters; you must specify a path and epoch"
            self.load_from_saved(path, epoch)
            self.save_dir = path
        
        self.model.to(device)

        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        self.n_class = n_class
        self.input_size = input_size
        self.layer_dims = list(zip([input_size] + n_neurons, n_neurons + [n_class]))

        # if the variance is set, we initialize a neural network with that variance
        if vary:
            weight_init_vary = nn_mod.create_weights_function(vary)
            self.model.apply(weight_init_vary)
            self.var = [vary] * len(n_neurons)
        else:
            self.var = np.divide(np.ones(len(n_neurons)), [i**2 for i in n_neurons])

        # all the rest of the features
        # set during training
        self.n_epochs = None
        self.epoch_history = None
        self.val_history = None
        self.train_history = None
        self.spectrum_history = None
        self.weights = None

        self.train_loader = None

        # set with effective dimensions
        self.normed_spectra = None
        self.adj_init_spectra = None
        self.rank_cutoffs = None
        self.effective_dimensions = None

        # activations
        self.activation_covs = None
        self.activation_spectrum = None

        # weights
        self.weight_covs = None
        self.weight_spectrum = None

    
    def load_from_saved(self, path, epoch):
        filename = f'{path}/{epoch}.pt'
        self.model = torch.load(filename)
        print(f'Model successfully loaded')
        return
    
    def set_train_loader(self, loader, overwrite=False):
        if (not self.train_loader) or (overwrite and loader):
            self.train_loader = loader
        else:
            print(f'warning: train loader will be overwritten. reenter command with overwrite=True if this is intended')

    def evaluate_model(self, test_loader):
        acc = train_network.evaluate_model(test_loader)

        return acc

    def get_spectrum(self):
        shh = []
        vvh = []
        cvh = []
        spectrum_history, _, _ = train_network.update_spectrum(self.model, shh, vvh, cvh)
        return spectrum_history

    def get_weights(self):
        weight_list = []
        for layer in self.model.layers:
            weight_list.append(layer.weight.numpy(force=True))

        self.weights = weight_list
        return weight_list

    def get_weight_covs(self):
        _ = self.get_weights()

        cov_layers = []
        for i in range(len(self.weights)-1):
            # print(i)
            lay_weights = self.weights[i]
            lay_neur = self.n_neurons[i]
            cov = lay_weights @ lay_weights.transpose() * (1/lay_neur)
            cov_layers.append(torch.from_numpy(cov))

        self.weight_covs = cov_layers
        return cov_layers

    def get_weight_spectrum(self):
        self.get_weight_covs()
        weight_spec = []
        for cov in self.weight_covs:
            vals, vecs = torch.linalg.eigh(cov)
            vals, vecs = vals.flip(-1), vecs.flip(-1)
            weight_spec.append(vals)
        self.weight_spectrum = weight_spec
        return

    def get_activations(self, dataloader, layers):
        # Version of get_activations which treats spatial dimensions as additional batch dimensions.
        get_acts = lambda *args: [align.space_to_batch(act) for act in align.get_activations(*args)]

        acts = []
        for x, _ in tqdm(dataloader, desc="Computing activations"):
            x = x.to(device)
            activations1 = get_acts(x, layers, self.model)
            acts.append(activations1)

        #self.activations = acts
        return acts

    def get_activation_covs(self, dataloader, layers):
        act_cov_layers = align.compute_activation_covariances(dataloader, layers, self.model)
        #act_cov_layers = [cov.detach().numpy() for cov in act_cov_layers]
        self.activation_covs = act_cov_layers.copy()
        return act_cov_layers

    def get_activation_spectrum(self):
        if self.activation_covs is None:
            self.get_activation_covs(self.train_loader, range(1, self.n_layers+1))

        act_spectra = []
        act_bases = []
        for cov in self.activation_covs:
            vals, vecs = torch.linalg.eigh(cov)
            # reverse and transpose
            #vals, vecs = np.flip(vals), np.flip(vecs)
            vals, vecs = vals.flip(-1), vecs.flip(-1)
            # vecs = vecs.T
            act_spectra.append(vals)
            act_bases.append(vecs)

        self.activation_spectrum = act_spectra
        self.activation_basis = [basis.T for basis in act_bases]

        return act_spectra

    def train(self, train_loader, val_loader, n_epochs, grain=5, ep_grain=2, 
              save=False):
        self.train_loader = train_loader
        e_list, val_hist, train_hist, spec_hist = train_network.train_model(self.model, train_loader, val_loader,
                                                                            n_epochs, grain=grain, ep_grain=ep_grain,
                                                                            save=save, savepath=self.save_dir)

        # setting the appropriate features
        if self.n_epochs is not None:
            self.n_epochs += n_epochs
        else:
            self.n_epochs = n_epochs

        # setting the appropriate features
        self.epoch_history = e_list
        self.val_history = val_hist
        self.train_history = train_hist

        # reshaping the spectrum history
        # this makes it so that each entry is a layer, and each entry in the layer is the spectrum corresponding to an
        # epoch checkpoint that happened
        spectra_lay = []
        for i in range(len(self.n_neurons)):
            layer = [hist[i] for hist in spec_hist]
            spectra_lay.append(layer)
        self.spectrum_history = spectra_lay

        self.get_weights() # setting the weights
        print(f'Model training complete!')

    def get_effective_dimensions(self, tail_match='mp', rankslist=None, scale='log', clip=None):
        """
        :param tail_match: string either 'mp' or 'rankspace'
        :return:
        """
        if not (tail_match == 'mp' or tail_match == 'rankspace'):
            raise Exception('invalid tail match regime. please enter either "mp" for marchenko-pastur or "rankspace" '
                            'to use the corresponding rank selection regime')
        # force-matching the tails
        diffed_specs, new_init_specs, rank_bound_list = eff_dim.match_spectrum_tails_regime(self, tail_match=tail_match,
                                                                                            rankslist=rankslist,
                                                                                            spacescale=scale,
                                                                                            clip=clip)

        # setting the features
        self.normed_spectra = diffed_specs
        self.adj_init_spectra = new_init_specs
        self.rank_cutoffs = rank_bound_list

        # getting the effective dimensionality
        effective_dimensions = eff_dim.effective_dimensionality(self)

        self.effective_dimensions = effective_dimensions

        print(f'Effective dimensions calculated')

    def plot(self, plotlist=('rel'), scale='log', layer=None, quantity=None, save_fig=False, xmax=None,
             saveadd=''):
        """

        :param plotlist: list of plots to create. Options are 'rel_eds', 'spec', 'rel', and 'acc'
        :param scale:
        :param layer:
        :param quantity:
        :param save_fig:
        :param xmax:
        :return:
        """
        if 'rel_eds' in plotlist:
            plotter.plot_relative_spectrum_history_eds(self, scale=scale, save_fig=save_fig, xmax=xmax, saveadd=saveadd)

        if 'spec' in plotlist:
            plotter.plot_spectrum(self, scale=scale, save_fig=save_fig, saveadd=saveadd)

        if 'rel' in plotlist:
            plotter.plot_spectrum_normed(self, scale=scale, save_fig=save_fig, xmax=xmax, saveadd=saveadd)

        if 'acc' in plotlist:
            plotter.plot_accuracy(self, save_fig=save_fig)

        if 'spec_sing' in plotlist:
            plotter.plot_spectrum_single(self, quantity, layer, scale=scale, save_fig=save_fig)

        return


########################
### HELPER FUNCTIONS ###
########################

def truncate_matrix_svd(matrix, rank):
    """

    :param matrix: array-like
    :param rank:    the rank to truncate at
    :return:
    """
    # first, get SVD factorization
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    U, S, Vh = U.flip(-1), S.flip(-1), Vh.flip(-1)

    # now, get the new diagonal
    high_val = S[rank - 1]
    new_S = torch.where(S < high_val, 0, S)

    # calculate the truncated matrix
    truncated = U @ torch.diag(new_S) @ Vh
    return truncated
