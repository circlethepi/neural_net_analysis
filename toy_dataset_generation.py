import numpy as np
import pickle
from sklearn.datasets import make_swiss_roll
from scipy import stats

"""
Generate various datasets to test the dimensionality estimation of the framework
"""

def generate_hypercube_dataset(n_points, n_dimensions):
    sample = np.random.randn(n_points, n_dimensions).tolist()
    #print(len(sample), len(sample[0]))
    return sample


def generate_swiss_roll_dataset(n_points):
    sample, _ = make_swiss_roll(n_points)
    #print(len(sample),  len(sample[0]))
    return sample


def generate_cauchy_dataset(n_points, n_dimensions):
    center = np.zeros(n_dimensions)
    shape = np.identity(n_dimensions)
    sample = stats.multivariate_t(center, shape, df=1).rvs(size=n_points).tolist()
    #print(len(sample), len(sample[0]))
    return sample


def generate_hypersphere_surface_dataset(n_points, n_dimensions=2):
    sample = np.random.randn(n_points, n_dimensions)
    sample /= np.linalg.norm(sample, axis=0)
    sample = sample.tolist()
    #print(len(sample), len(sample[0]))
    return sample


def generate_multivariate_normal(n_points, n_dimensions=2, mean=None):
    mean = mean if mean else np.zeros(n_dimensions)
    cov_mat = np.identity(n_dimensions)
    sample = stats.multivariate_normal(mean, cov=cov_mat).rvs(size=n_points).tolist()
    #print(len(sample), len(sample[0]))
    return sample


def convert_array_datasets_to_dataloaders(dataset, n_classes):
    """

    :param dataset:     list(array) : the dataset to convert into a torch dataloader
    :param n_classes:   int         : the number of class options to randomly assign to the samples in the dataset
    :return: loader :   torch.DataLoader
    """

    loader = 0
    return loader


def pickle_dataset(pickle_name):
    filename = f'{pickle_name}.pkl'

    print(f'dataset successfully pickled at {pickle_name}')
    return

