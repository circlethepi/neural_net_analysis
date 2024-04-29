import pandas as pd
import numpy as np
import torch

import class_splitter as cs
import network_similarity as sim
import matplotlib.pyplot as plt
from sklearn import manifold

#################################
#
#################################


class class_distance_collection:

    def __init__(self, n_layers, datarows=None, filenames=None, classes=None):
        #  create the header list
        class_names = ['class1', 'class2']
        layer_names = []
        for i in range(1, n_layers+1):
            layer_names.append(f'layer {i}')
        name_list = class_names + layer_names

        self.layer_list = layer_names

        if filenames:
            act_filename, way_filename = filenames
            # read in the files as dataframes
            self.filenames = (act_filename, way_filename)
            self.dfa = pd.read_csv(act_filename, header=None, names=name_list)
            self.dfw = pd.read_csv(way_filename, header=None, names=name_list)

        if datarows:
            self.filenames = None
            self.dfa = pd.DataFrame(datarows[0], columns=name_list)
            self.dfw = pd.DataFrame(datarows[1], columns=name_list)

        print(self.dfa.columns)

        self.classes = sorted(set(list(self.dfa['class1']) + list(self.dfa['class2'])))

        # convert each of the distances into complex numbers
        for df in (self.dfa, self.dfw):
            for col in layer_names:
                new_col = df[col].apply(dist_conv)
                df[col] = new_col

        # # normalize so that each class distance is 0 for itself in each df
        # for df in (self.dfa, self.dfw):
        #     for class_name in self.classes:
        #         # get the base row
        #         class_row = df[(df['class1'] == df['class2']) & (df['class1'] == class_name)]
        #         # Get the layer values from that row
        #         class_layer_values = class_row[self.layer_list].values[0]
        #
        #         # subtract the base values from the corresponding values where class1 == class_name
        #         class1_rows = df[df['class1'] == class_name]
        #         df.loc[class1_rows.index, self.layer_list] -= class_layer_values

        # convert each of the centered distances into real distances
        for df in (self.dfa, self.dfw):
            for col in layer_names:
                new_col = df[col].apply(dist_real)
                df[col] = new_col

        ################################
        # set the rest of the attributes
        self.activation_distance_mat = None
        self.weight_distance_mat = None

    def create_distance_matrices(self):
        dist_mat_list = []
        # for each layer
        for layer in self.layer_list:
            # get the distance matrix
            layer_mat = df_to_distmat(self.dfa, 'class1', 'class2', meas_col=layer)
            dist_mat_list.append(layer_mat)
            #print(layer_mat)

        self.activation_distance_mat = dist_mat_list.copy()
        print('Set Activation distance matrix')

        dist_mat_list = []
        # for each layer
        for layer in self.layer_list:
            # get the distance matrix
            layer_mat = df_to_distmat(self.dfw, 'class1', 'class2', meas_col=layer)
            dist_mat_list.append(layer_mat)

        self.weight_distance_mat = dist_mat_list.copy()
        print('Set Weight distance matrix')
        return

    def create_distance_map(self):
        frames_dict = {'activations': self.activation_distance_mat,
                       'weights': self.weight_distance_mat}

        for i in range(len(self.layer_list)):
            for quantity in ('activations', 'weights'):
                df = frames_dict[quantity][i]

                heatmap_title = f'{quantity} similarity heatmap\nLayer{i+1}'

                coords = distmat_to_map(df, heatmap_title)
                title = f'{quantity} class map\nLayer {i+1}'

                plot_map(coords, self.classes, title)

        return

    def get_sim_scores(self):
        a_max = self.dfa.max()


        scores = []
        return scores

    def plot_distmats(self):

        return


def dist_conv(x):
    """

    :param x: string : of a complex number (output from network_similarity.bw_dist)
    :return: complex float :
    """
    # return complex(x)
    return float(x)


def dist_real(x):
    return np.abs(x)


def df_to_distmat(df, lab_col_1, lab_col_2, meas_col):
    idx = sorted(set(df[lab_col_1]).union(df[lab_col_2]))
    #print(set(df[lab_col_1]).union(df[lab_col_2]))

    df.reset_index(inplace=True, drop=True)

    dfr = (df.pivot(index=lab_col_1, columns=lab_col_2, values=meas_col).fillna(0, downcast='infer'))
           #.reindex(index=idx, columns=idx)
           #.fillna(0, downcast='infer'))

    dfr += dfr.T

    return dfr


def distmat_to_map(df, title):
    # get the coordinates
    adist = df.to_numpy()
    names = df.columns.tolist()
    ticks = list(range(len(names)))

    amax = np.amax(adist)
    adist /= amax

    mds = manifold.MDS(n_components=2, dissimilarity="precomputed")
    mds.fit(adist)

    coords = mds.embedding_

    sims = np.ones(np.shape(adist)) - adist

    # plot the normalized distances as a heatmap
    fig = plt.figure(figsize=(4,4))
    #ax.imshow(adist, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    mappy = plt.imshow(sims, cmap='binary', vmin=0, vmax=1,
                       interpolation='nearest')
    plt.colorbar(mappy, fraction=0.045)

    plt.xticks(ticks=ticks, labels=names)
    plt.yticks(ticks=ticks, labels=names)

    plt.title(title)
    plt.show()
    #path = "/Users/mnzk/Documents/40-49. Research/42. Nets/42.97. Library/image_hold/"
    #plt.savefig()

    return coords


def plot_map(coords, classes, title):
    """

    :param coords: output of distmat_to_map; an array where the first row are the x values and the second row are the
                                            y values
    :param classes: list of classes of length (len(distmat))
    :param title:
    :return:
    """
    fig = plt.figure(figsize=(5,5))
    plt.subplots_adjust(bottom=0.1)

    plt.scatter(coords[:, 0], coords[:, 1], marker='o')

    for label, x, y, in zip(classes, coords[:, 0], coords[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(-12, -12),
                     textcoords='offset points', ha='right', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.2', fc='blue', alpha=0.2),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.title(title)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    #plt.xticks([])
    #plt.yticks([])

    plt.show()
    return

