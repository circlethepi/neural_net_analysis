# Neural Net Analysis

> Dear reader, I apologize in advance for the disorganized-ness of my code. As this project is ongoing, I have not yet taken the opportunity to clean and package everything. However, in this markdown I have included instructions on how to reproduce the majority of the figures included in my Probabilistic Machine Learning final project. 

A quick summary of what each file does:
- **neural_network** defines a simple linear neural network object, loads datasets, and creates unmodified dataloaders. I can also create multiple training dataloaders to create partitions of the training data. 
- **train_network** contains the train procedure for a linear fully connected network, including the collection of the weight spectrum of the network at intervals throughout training
- **spectral_analysis** defines a `spectrum_analysis` object, which does much of the heavy lifting when it comes to analyzing fully connected networks. It wraps around a neural network and contains functions to analyze the spectrum of both the weights and activations of the model. It can also calculate the effective dimensionality. It can plot various figures related to the accuracy, weight spectrum, and dimensionality.
- **plotting_networks** contains all of the plotting functions used by the `spectrum_analysis` object to plot various features. 
- **effective_dimensions** contains the functions required to calculate the effective dimensions of a network under two different tail match/subtract regimes. The first is using the Marchenko-Pastur bound, and the second is by setting a range of ranks over which the spectrum tails should be matched to.  
- **network_similarity** defines a `network_comparison` object, which does exactly as implied and compares two networks. It calculates the alignment matrices for networks, the aligned and unaligned cosine similarities between bases of the matrices, plots those similarities and can calculate a similarity and a distance metric between the two networks. 
- **alignment** contains the functions needed in order to calculate the alignment matrices between two models.
- **distance_mapping** creates a 2D map of networks from pairwise distances
- **class_splitter** makes more complicated dataloaders. It can create dataloaders that are subsets of the full training dataset and create loaders containing perturbed data. Currently, the perturbations included are the covering of the image with colored columns or rows.