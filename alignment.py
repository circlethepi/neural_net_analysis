# import
import numpy as np
from utils import AverageMeter
from tqdm import tqdm
#from tqdm.notebook import tqdm
import torch
import scipy

from matplotlib import pyplot as plt
from utils import set_torch_device

#######################
# Getting Activations #
#######################


device = set_torch_device()


def get_activations(x, layers, model):
    """
    Returns the hidden activations of a model.
    :param x: input to use, tensor of shape (B, C, [N, N])
    :param layers: list of integers (j corresponds to output of j-th layer, 0 corresponds to input of model)
    :param model: model to use (should be Sequential)
    :return: list of saved activations of same length as layers
    """
    saved_activations = []

    def hook(self, inputs, output):  # inputs is a tuple, we assume it is of length 1
        saved_activations.append(inputs[0])

    # Register hooks to save activations of chosen layers.
    for layer in layers:
        model.layers[layer].register_forward_hook(hook)

    # Forward of model: hooks will be run and activations will be saved.
    _ = model(x)

    # Clear hooks.
    for layer in layers:
        model.layers[layer]._forward_hooks.clear()

    return saved_activations


####################
# reshaping inputs #
####################
def space_to_batch(x):
    """ (B, C, [M, N]) to (B[MN], C). """
    if x.ndim == 4:
        x = x.permute(0, 2, 3, 1)  # (B, M, N, C)
        x = x.reshape((-1, x.shape[-1]))  # (BMN, C)
    return x


#########################
# Computing Covariances #
#########################
def compute_activation_covariances(loader, layers, model1, model2=None):
    """ Compute the (cross-)covariance of hidden activations at several layers of one or two models.
    :param loader: data loader to use
    :param layers: list of integers (j corresponds to output of j-th layer, 0 corresponds to input of model)
    :param model1: model to use (should be Sequential)
    :param model2: optional model for a cross-covariance (if None, compute the self-covariance of model1)
    :return: list of computed covariances (C1, C2), of same length as layers
    """
    meters = [AverageMeter() for _ in layers]

    # Version of get_activations which treats spatial dimensions as additional batch dimensions.
    get_acts = lambda *args: [space_to_batch(act) for act in get_activations(*args)]

    for x, _ in loader:
        x = x.to(device)
        activations1 = get_acts(x, layers, model1)
        activations2 = activations1 if model2 is None else get_acts(x, layers, model2)

        for i, (act1, act2) in enumerate(zip(activations1, activations2)):
            cov = act1.T @ act2  # (C1, C2), sum of outer products over the batch
            meters[i].update(val=cov, n=act1.shape[0])

    return [meter.avg() for meter in meters]


###################################
# Computing Alignments for Models #
###################################
def compute_alignments(loader, layers, model1, model2):
    """
    Aligns Model 2 to model 1
    """
    # getting the layer covariances for each model (and each layer)
    # model1_layer_covs = compute_activation_covariances(loader, layers, model1)
    # model2_layer_covs = compute_activation_covariances(loader, layers, model2)

    # getting the cross covariances
    cross_covs = compute_activation_covariances(loader, layers, model1, model2)
    # print('cross cov sizes')
    # for cov in cross_covs:
    #    print(list(cov.size()))

    # getting the alignments
    aligns = []
    # r_squareds = []
    for j in range(len(layers)):
        cross_cov = cross_covs[j]
        u, s, vh = torch.linalg.svd(cross_cov, full_matrices=False)

        # getting the explained variances
        # explained = torch.sum(s)
        # total = torch.sqrt(torch.trace(model1_layer_covs[j]) * torch.trace(model2_layer_covs[j]))
        # r_squared = explained / total
        # print(f'Layer {layers[j]}: {100 * r_squared.item():.1f}% of variance explained by alignment')

        align = u @ vh
        aligns.append(align)
        #r_squareds.append(r_squared)

    return aligns#, r_squareds


#######################
# Aligning the Models #
#######################
def aligning_models(loader, layers, model1, model2, clip=50, eff_dims=False, names=None, metric=False, bw=False):
    # setting the graphing tic marks
    step = int(10 ** np.floor(np.log10(int(np.floor(clip / 5)))))
    # step = int(np.floor(clip/5))
    ticks = np.array(range(0, clip, step))

    # layers is a list of layer indices, so the actual "layers" are j + 1 for each index j
    align_list, r2s = compute_alignments(loader, layers, model1.model, model2.model)
    # print('align sizes')
    # for a in align_list:
    #    print(list(a.size()))

    # get the activation covs
    act_cov1 = compute_activation_covariances(loader, layers, model1.model)
    act_cov2 = compute_activation_covariances(loader, layers, model2.model)

    _ = model1.get_weight_spectrum()
    _ = model2.get_weight_spectrum()

    # get the weight covs
    weight_cov1 = model1.get_weight_covs()
    weight_cov1 = [weight_cov1[i - 1] for i in layers]
    weight_cov2 = model2.get_weight_covs()
    weight_cov2 = [weight_cov2[i - 1] for i in layers]

    # get the weight spectrum (only contains the layers that we are looking at)
    weight_spec1 = [model1.weight_spectrum[i - 1] for i in layers]
    weight_spec2 = [model2.weight_spectrum[i - 1] for i in layers]

    # get the activation spectrum
    act_spec1 = []
    act_spec2 = []

    # getting the weight eigenvectors and activation eigenvectors
    weight_eigenvectors1 = []
    weight_eigenvectors2 = []

    activation_eigenvectors1 = []
    activation_eigenvectors2 = []
    for j in range(len(layers)):
        layer_ind = layers[j]
        # weights
        if layer_ind != 0:
            w1 = torch.from_numpy(model1.weights[layer_ind - 1])
            w2 = torch.from_numpy(model2.weights[layer_ind - 1])
            # print('weight sizes')
            # print(list(w1.size()))

            u1, s1, vh1 = torch.linalg.svd(w1, full_matrices=False)
            u2, s2, vh2 = torch.linalg.svd(w2, full_matrices=False)

            weight_eigenvectors1.append(vh1)
            weight_eigenvectors2.append(vh2)

        # activations
        a1 = act_cov1[j].detach().numpy()
        a2 = act_cov2[j].detach().numpy()
        # get the eigendecomp
        vals1, vecs1 = np.linalg.eigh(a1)
        vals2, vecs2 = np.linalg.eigh(a2)
        # sort in desc order and transpose
        vals1, vecs1 = torch.from_numpy(vals1).flip(-1), torch.from_numpy(vecs1).flip(-1)
        vecs1 = vecs1.T
        vals2, vecs2 = torch.from_numpy(vals2).flip(-1), torch.from_numpy(vecs2).flip(-1)
        vecs2 = vecs2.T

        activation_eigenvectors1.append(vecs1)
        activation_eigenvectors2.append(vecs2)
        act_spec1.append(vals1)
        act_spec2.append(vals2)

    # extracting effective dimensions if indicated
    if eff_dims:
        dims = [mod.effective_dimensions for mod in [model1, model2]]

    # adding names if they are given
    name1 = names[0] if names else "Network 1"
    name2 = names[1] if names else "Network 2"

    # metric/distance container
    unaligned_acts = []
    aligned_metric_acts = []
    # sim container
    unaligned_sims_acts = []
    aligned_sim_acts = []

    # now compute and plot cossim for activations
    for aligned in [False, True]:
        for j in range(len(layers)):
            if aligned:
                align_mat = align_list[j]
                # print('matrix shapes')
                # print('activation sizes: ', list(activation_eigenvectors1[j].size()), list(align_mat.size()), list(activation_eigenvectors2[j].T.size()))
                sim = torch.abs(activation_eigenvectors1[j] @ align_mat @ activation_eigenvectors2[j].T)
                aligned_sim_acts.append(sim)
                if metric:
                    # aligned_metric_acts.append(get_sim_metric(torch.clone(sim), clip))
                    aligned_metric_acts.append(
                        get_sim_metric(act_spec1[j], activation_eigenvectors1[j].detach().numpy(),
                                       act_spec2[j], activation_eigenvectors2[j].detach().numpy(),
                                       align_mat.detach().numpy(), clip))
                if bw:
                    aligned_vecs2 = activation_eigenvectors2[j] @ align_mat.T
                    # print(f'shape aligned vecs = {np.shape(aligned_vecs2)}')
                    aligned_cov_2 = aligned_vecs2.T @ torch.diag(act_spec2[j]) @ aligned_vecs2
                    # print(f'spec shape = {np.shape(act_spec2[j])}\nshape aligned covs = {np.shape(aligned_cov_2)}')
                    aligned_metric_acts.append(bw_dist(act_cov1[j].detach().numpy(), aligned_cov_2.detach().numpy()))
            else:
                sim = torch.abs(activation_eigenvectors1[j] @ activation_eigenvectors2[j].T)
                unaligned_sims_acts.append(sim)
                if bw:
                    unaligned_acts.append(bw_dist(act_cov1[j].detach().numpy(), act_cov2[j].detach().numpy()))
                # if metric:
                # getting similarity of unaligned (this will be 0 by our definition)
                #    unaligned_acts.append(get_sim_metric(torch.clone(sim), clip))

            # print('activation sim size   : ', list(sim.size()))

            # plotting
            with (torch.no_grad()):
                fig = plt.figure(figsize=(8, 8))
                mappy = plt.imshow(sim[:clip, :clip], cmap='binary', vmin=0, vmax=1)
                plt.colorbar(mappy, fraction=0.045)
                plt.ylabel(f'Rank - {name1}')
                plt.xlabel(f'Rank - {name2}')
                al_title = 'aligned ' if aligned else ''
                second = (f'\n{100 * r2s[j].item():.1f}% variance explained by alignment') if aligned else ""
                if metric:
                    second += f'\nSimilarity Score: {aligned_metric_acts[-1]:.2E}' if aligned else ''  # (f'\nSimilarity Score: '
                    # f'{unaligned_acts[-1]:.2f}')
                if bw:
                    second += f'\nBW-Distance: {aligned_metric_acts[-1]:.2f}' if aligned else f'\nBW-Distance: {unaligned_acts[-1]:.2f}'

                plt.title(f'Cosine Similarity of {al_title}activation eigenvectors, layer {layers[j]}{second}')

                # if eff_dims:
                #    plt.vlines(dims[0][layers[j]][-1], 0, clip, colors='r', linestyles=':')
                #    plt.hlines(dims[1][layers[j]][-1], 0, clip, colors='b', linestyles=':')

                plt.xticks(ticks=ticks - 1, labels=[str(t) for t in ticks])
                plt.yticks(ticks=ticks - 1, labels=[str(t) for t in ticks])

    # ways dist/metric
    unaligned_ways = []
    aligned_metric_ways = []
    # sim containers
    unaligned_sim_ways = []
    aligned_sim_ways = []
    # Now, do it for the weights
    for aligned in [False, True]:
        for j in range(len(layers)):
            if aligned and j > 0:
                align_mat = align_list[j - 1]  # -1]
                # print('weight matrix sizes: ', list(weight_eigenvectors1[j].size()), list(align_mat.size()), list(weight_eigenvectors2[j].T.size()))
                sim = torch.abs(weight_eigenvectors1[j] @ align_mat @ weight_eigenvectors2[j].T)
                aligned_sim_ways.append(sim)
                if metric:
                    # aligned_metric_ways.append(get_sim_metric(torch.clone(sim), clip))
                    aligned_metric_ways.append(get_sim_metric(weight_spec1[j], weight_eigenvectors1[j].detach().numpy(),
                                                              weight_spec2[j], weight_eigenvectors2[j].detach().numpy(),
                                                              align_mat.detach().numpy(), clip))
                if bw:
                    aligned_weights2 = weight_eigenvectors2[j] @ align_mat.T
                    aligned_cov2 = aligned_weights2.T @ torch.diag(
                        torch.from_numpy(weight_spec2[j].copy())) @ aligned_weights2
                    aligned_metric_ways.append(bw_dist(weight_cov1[j], aligned_cov2.detach().numpy()))
            else:
                sim = torch.abs(weight_eigenvectors1[j] @ weight_eigenvectors2[j].T)
                unaligned_sim_ways.append(sim)

                if aligned and metric:
                    al = np.zeros(np.shape(align_list[0]))
                    al[:max(np.shape(align_list[0])), :max(np.shape(align_list[0]))] = np.ones(
                        max(np.shape(align_list[0])))

                    # print(f'SHAPES\nweight spec 1: {np.shape(weight_spec1[j])}\n'
                    #      f'weight eigs 1: {np.shape(weight_eigenvectors1[j].detach().numpy())}\n'
                    #      f'weight spec 2: {np.shape(weight_spec2[j])}\n'
                    #      f'weight eigs 2: {np.shape(weight_eigenvectors2[j].detach().numpy())}\n'
                    #      f'align mat too: {np.shape(al)}')

                    sim_here = get_sim_metric(weight_spec1[j], weight_eigenvectors1[j].detach().numpy(),
                                              weight_spec2[j],
                                              weight_eigenvectors2[j].detach().numpy(), al, clip)
                    aligned_metric_ways.append(sim_here)
                elif bw:
                    bww = bw_dist(weight_cov1[j], weight_cov2[j])
                    if aligned:
                        aligned_metric_ways.append(bww)
                    else:
                        unaligned_ways.append(bww)

                # elif metric:
                #    unaligned_ways.append(sim_here)

            # print('weight sim size   : ', list(sim.size()))
            with (torch.no_grad()):
                # plotting
                fig = plt.figure(figsize=(8, 8))
                mappy = plt.imshow(sim[:clip, :clip], cmap='binary', vmin=0, vmax=1,
                                   interpolation='nearest')  # vmin=0, vmax=1,
                plt.colorbar(mappy, fraction=0.045)
                plt.ylabel(f'Rank - {name1}')
                plt.xlabel(f'Rank - {name2}')
                al_title = 'aligned ' if aligned else ''

                if metric:
                    sim_title = f'\nSimilarity Score: {aligned_metric_ways[-1]:.2E}' if aligned else ''  # f'\nSimilarity Score: {unaligned_ways[-1]:.2f}'
                elif bw:
                    print('weight aligned bw: ', aligned)
                    sim_title = f'\nBW-Distance: {aligned_metric_ways[-1]:.2f}' if aligned else (f'\nBW-Distance: '
                                                                                                 f'{unaligned_ways[-1]:.2f}')
                else:
                    sim_title = ''

                plt.title(f'Cosine Similarity of {al_title}weight eigenvectors, layer {layers[j]}{sim_title}')

                if eff_dims:
                    # line 1
                    ed1y = [dims[0][layers[j] - 1][-1], dims[0][layers[j] - 1][-1]]
                    ed1x = [0, clip - 1]

                    ed2y = [0, clip - 1]
                    ed2x = [dims[1][layers[j] - 1][-1], dims[1][layers[j] - 1][-1]]

                    # then plot
                    plt.plot(ed1x, ed1y, color='r', linestyle='dashed', linewidth=3, label=f'{name1}')
                    plt.plot(ed2x, ed2y, color='b', linestyle='dashed', linewidth=3, label=f'{name2}')
                    plt.legend(title=f'effdims after {model1.epoch_history[-1]} epochs')

                plt.xticks(ticks=ticks - 1, labels=[str(t) for t in ticks])
                plt.yticks(ticks=ticks - 1, labels=[str(t) for t in ticks])

    return unaligned_sims_acts, aligned_sim_acts, unaligned_sim_ways, aligned_sim_ways
    # list(np.array(aligned_metric_acts)), list(np.array(aligned_metric_ways))


##########################
# Plotting Cossim Matrix #
##########################

def plot_cossim_mat(mat1, mat2, names=None, clip=50):
    # setting the names of the networks
    name1 = names[0] if names else "Network 1"
    name2 = names[1] if names else "Network 2"

    # plotting

    return


#############################
# Aligned Similarity Metric #
#############################

def truncate_mat(clip, vals, vecs1, vecs2h=None):
    '''

    :param clip: int            the rank to clip to
    :param vals: torch.tensor   either the eigenvalues or the singular values of the matrix
    :param vecs1: torch.tensor  the left singular vectors or the eigenvectors of the matrix
    :param vecs2h: torch.tensor the right singular vectors of the matrix. If none, these will be set to the eigenvectors
                                transposed
    :return: new_mat: torch.tensor       the truncated matrix (rank clip approximation)
    '''
    # clipping the vals
    # new_vals = [vals[i] if i <= clip else 0 for i in range(len(vals))]
    highest = vals[clip - 1]
    new_vals = torch.where(vals < highest, 0, vals)

    # doing the vectors
    if vecs2h is None:
        vecs2h = torch.clone(vecs1).T

    # making the diag matrix
    new_diag = torch.zeros((vecs1.size()[1], vecs2h.size()[0]))
    new_diag[:len(new_vals), :len(new_vals)] = torch.diag(new_vals)

    # print(f'truncate shapes\n'
    #      f'vecs 1: {np.shape(vecs1)}\n'
    #      f'vecs 2: {np.shape(vecs2h)}\n'
    #      f'diag  : {np.shape(new_diag)}')

    # reconstructing the truncated matrix
    new_mat = vecs1 @ new_diag @ vecs2h

    return new_mat


def mat_cossim_covs(mat1, mat2):
    # num = np.linalg.norm(np.array(scipy.linalg.sqrtm(mat1)) @  np.array(scipy.linalg.sqrtm(mat2)), ord='nuc')
    mat1_12 = scipy.linalg.sqrtm(mat1)
    mat2_12 = scipy.linalg.sqrtm(mat2)

    mult = np.array(mat1_12) @ np.array(mat2_12)

    u, s, vt = np.linalg.svd(mult)
    num = np.sum(s)

    denom = np.sqrt(np.trace(mat1) + np.trace(mat2))
    print('numerator = ', num, '\ndenominator =', denom, )
    print('sim = ', num / denom)

    return num / denom


def bw_dist(mat1, mat2):
    print('mat2 shape = ', np.shape(mat2))
    # a standard practice to find the square root of a matrix is to use the eigen decomposition and then take the square
    # root of the eigenvalues
    mat1_12 = np.array(scipy.linalg.sqrtm(mat1))
    mult = mat1_12 @ mat2 @ mat1_12
    mult_12 = np.array(scipy.linalg.sqrtm(mult))
    trace1 = np.trace(mat1)
    trace2 = np.trace(mat2)
    tracemult = 2 * (np.trace(mult_12))

    bw_dist = trace1 + trace2 - tracemult
    bw_dist = np.sqrt(bw_dist)

    return bw_dist


def floor_sim(sim, floor=0.3):
    new = sim.copy()
    new[new < floor] = 0
    return (np.sum(new) / len(new)) * 100


def get_sim_metric(vals1, vecs1, vals2, vecs2, align_mat, clip):
    """

    """
    c1 = truncate_mat(clip, vals1, vecs1)
    oc1ot = truncate_mat(clip, vals2, vecs2)
    c2 = align_mat @ oc1ot @ align_mat.transpose()

    num = mat_cossim_covs(c1, c2) - mat_cossim_covs(c1, oc1ot)
    # denom = mat_cossim_covs(c1, c1) - mat_cossim_covs(c1, oc1ot)
    # print(denom)

    return num  # /denom


####################################
# Plotting Sim Measurement Results #
####################################

def plot_sim_results(results):
    return
