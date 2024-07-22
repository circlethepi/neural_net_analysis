"""Package Import"""
import numpy as np

import alignment as align
import neural_network as nn_mod
import perturbation as pert
import spectral_analysis as spec
from utils import *

"""
TO-DO

"""

class ActivationCollector():
    """
    Collects activations from a model
    """

    def __init__(self, path, epoch):
        """
        Takes in the path and epoch to find a model save file
        """
        specmod = spec.spectrum_analysis([], load=True, 
                                         path=path, epoch=epoch)
        
        self.model = specmod.model
        del specmod

        return


class ActivationCollection():
    """
    """

    def __init__(self):
        return


class ActivationPlotter():
    """
    """

    def __init__(self):
        return