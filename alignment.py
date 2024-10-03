# import
import numpy as np
from utils import AverageMeter
from tqdm import tqdm
#from tqdm.notebook import tqdm
import torch
import scipy
import time

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
    # print(f'getting activations for layers {layers}')

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
        # turns out doing this contributes wayy more time than strictly necessary oop
        # putting back to see if tis fixes it 2024-10-02
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
        #print(f'Layer {layers[j]}: {100 * r_squared.item():.1f}% of variance explained by alignment')

        align = u @ vh
        aligns.append(align)
        # r_squareds.append(r_squared.item())
        #print(r_squared)

    return aligns#, r_squareds



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
