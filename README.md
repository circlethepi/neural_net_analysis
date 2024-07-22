# Neural Net Analysis

All models should be stored in a directory named for the model. The files inside the directory hold the saved models with `.pt` extensions, and should be named as the epoch after which the save file was created. 
> [!example] Example Save Configuration
> ``` 
> root
>  |- saved_models
>  		|- ModelName
> 			|- 0.pt
> 			|- 5.pt
> 			|- 10.pt
>			|- 50.pt
> ```


> [!note]
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

#### To create the stacked spectrum plots (see section 1.1)
1. load in train and val dataloaders of CIFAR-10 using `neural_network.get_dataloaders()`
2. Create a `spectral_analysis.spectrum_analysis` object with the desired number of layers and neurons. For a one-layer network, do 
```
model = spectral_analysis.spectrum_analysis([N_NEURONS])
```
3. Train the model for however long you want
```
model.train(train_loader, val_loader, n_epochs=4)
```
4. Get the weight spectrum and effective dimensionality
```
model.get_weight_spectrum()
model.get_effective_dimensions()
```
5. Plot the results
```
model.plot(['spec'])
```
6. To get the tail-matched plots with effective dimensions and accuracy, do
```
model.plot(['rel_eds'])
```

#### To create cosine similarity plots for two networks (see section 2 and 3)
1. Create two `spectral_analysis.spectrum_analysis` objects and train each of them as in the above tutorial. They should have the same architecture. You can use `class_splitter` to create different configurations of the same dataset if desired for training.
2. Create a `network_similarity.network_comparison` object:
```
netsim = network_similarity.network_comparison(MODEL1, MODEL2, names=(NAME1, NAME2))
```
If desired, you can also name the models.
3. Calculate the alignment between the networks
```
netsim.compute_alignments(DATALOADER, [LAYER_LIST])
```
This computes the alignment matrices for each layer in the `LAYER_LIST` with respect to the input data from `DATALOADER`
4. Compute the cosine similarities for the aligned and unaligned networks
```
netsim.compute_cossim()
```
5. plot the cosine similarity matrices
```
netsim.plot_sims(
	clips = [CLIPS],
	layers=None,  
	quantities=('activations', 'weights'),  
	alignments=[False, True],  
	plot_clip=None,  
	filename_append=None,  
	ed_plot=False)
)
```
- `CLIPS` is the list of ranks at which to clip the covariance matrices before comparing. 
- `layers` is the list of layers to plot. If `None`, it plots all layers
- `quantities` are the quantities to plot for each layer
- `aligments` are the values for whether to plot aligned or not for each quantity at each layer. By default, this is only `True`
- `plot_clip` is optional and will overlay lines at the ranks indicated
- `filename_append` is an additional tag to add to the filename when the figures are being saved
- `ed_plot` is a boolean on whether or not to overplot the effective dimensions of each network