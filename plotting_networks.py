# basics
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import utils


image_saveloc = f'image_hold'


def plot_relative_spectrum_history_eds(model, scale='log', save_fig=True, xmax=1000, saveadd=""):
    begin = 1
    epochs = model.epoch_history
    # print([k if isinstance(k, str) else "" for k in epochs])
    epochs = np.array([k if isinstance(k, int) else float(k) for k in epochs])
    # print(epochs)

    if not xmax:
        xmax = model.n_neurons

    for j in range(len(model.normed_spectra)):
        rel_spec = model.normed_spectra[j]
        eff_dim = model.effective_dimensions[j]

        print(len(epochs), len(model.normed_spectra))
        colors = plt.cm.Spectral(np.linspace(0, 1, len(epochs)))
        # colors = plt.cm.viridis(np.linspace(0, 1, len(epochs) +1))

        fig, ax = plt.subplots(figsize=(10, 5))

        ed_ys = []

        for i in range(begin, len(epochs)):
            xs = list(range(1, len(rel_spec[i-1]) + 1))
            ys = rel_spec[i-1]
            # plt.plot(xs, ys, label=f'epoch {epochs[i]}', color=colors[i])
            plt.plot(xs, ys, color=colors[i])

            # get the intersection of the relative spectrum and the effective dimensionality
            # if i < len(epochs):
            y_val = np.interp(eff_dim[i], xs, ys)
            ed_ys.append(y_val)

            # plot the effective dimensions intersections
            # test_accuracies = [int(model.test_history[i]*100) for i in range(1,len(model.test_history))]
        test_accuracies = model.val_history[begin:]
            # acc_colors = plt.cm.Greys(np.linspace(0,1,100))
        plt.plot(eff_dim[begin:], ed_ys, 'k--', label='Eff. Dim.')

        plt.tick_params(axis='both', which='both', labelsize=16)

            # colormaps: spring, coolwarm, Greys
        acc = ax.scatter(eff_dim[begin:], ed_ys, c=test_accuracies, cmap=plt.cm.spring, s=20, edgecolors="black",
                        label='Val Accuracy', zorder=10)
        
        # double colorbar
        divider = make_axes_locatable(ax)

        # accurcaies
        cax1 = divider.new_horizontal(size="5%", pad=0.2)
        fig.add_axes(cax1)
        cbar1 = fig.colorbar(acc, cax=cax1)
        cbar1.ax.tick_params(labelsize=16)
        cbar1.set_label('accuracy', fontsize=16)

        # epochs
        cax2= divider.new_horizontal(size="5%", pad=1.2, pack_start=True)
        fig.add_axes(cax2)
        epoch_colormap = mpl.colors.ListedColormap(colors[1:])
        bounds = np.linspace(0, 1, len(epochs))
        # norm = mpl.colors.LogNorm(vmin=min(epochs[epochs>0]), vmax=max(epochs))
        norm = mpl.colors.BoundaryNorm(bounds, epoch_colormap.N)
        cbar2 = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=epoch_colormap),
                             cax=cax2)
        cbar2.ax.tick_params(labelsize=16)
        cbar2.ax.yaxis.set_ticks_position('left')
        midpoints = 0.5 * (bounds[:-1] + bounds[1:])
        cbar2.set_ticks(midpoints)
        cbar2.set_ticklabels([int(k) if int(k)-k ==0 else f'{k:.2f}' for k in epochs[1:]])
        cbar2.set_label('epoch', fontsize=16)
        cbar2.ax.yaxis.set_label_position('left')

        ax.set_title(f'Spectrum evolution over epochs\n{model.n_neurons} neurons, Layer {j+1} of {model.n_layers}', fontsize=16)
        # setting the scale
        ax.set_xscale(scale)
        ax.set_yscale(scale)

        ax.set_xlabel('Rank', fontsize=16)
        ax.set_ylabel('eigenvalue', fontsize=16)

        #plt.xlim(1, xmax)
        #plt.xlim(min([min(rel_spec), min(eff_dim), 1]), xmax)
        ax.set_ylim(np.max([np.min(rel_spec), 10**(-5)]), np.max(rel_spec))
        # plt.ylim(np.min(rel_spec), np.max([np.max(rel_spec), model.var]))
        plt.tick_params(axis='both', which='both', labelsize=16)

        ax.legend(reverse=True, loc='upper right', fontsize=16)

        if save_fig:
            plt.savefig(f'{image_saveloc}/ED_rel_spec_hist_{model.n_neurons}_{model.n_epochs}_layer{j+1}{saveadd}.png')
                # files.download(f'ED_rel_spec_hist_{model.n_neurons}_{model.n_epochs}_layer{j+1}.png')

        plt.show()


def plot_spectrum(model, scale='log', save_fig=True, saveadd=""):
  epochs = model.epoch_history

  for j in range(len(model.spectrum_history)):
    spec_hist = model.spectrum_history[j]

    print(f'n epoch measures: {len(epochs)}\nn spectra recs: {len(spec_hist)}')
    colors = plt.cm.Spectral(np.linspace(0, 1, len(epochs) +1))
    #colors = plt.cm.viridis(np.linspace(0, 1, len(epochs) +1))


    fig = plt.figure(figsize = (10, 5))

    ed_ys = []

    for i in range(len(epochs)):
      # ranks
      xs = list(range(1, len(spec_hist[i]) + 1))
      # eigenvalues
      ys = spec_hist[i]
      # plot the ranks vs the eigenvalues
      plt.plot(xs, ys, label=f'epoch {epochs[i]}', color=colors[i])
      # get the intersection of the relative spectrum and the effective dimensionality
      #y_val = np.interp(model.effective_dimensions[i-1], xs, ys)
      #ed_ys.append(y_val)


      # add the baseline variance
    #plt.hlines(model.var, 1, len(ys), colors='k', linestyles = "dashed")#, label = f"init var = {model.var}")

    plt.title(f'Spectrum evolution over epochs \
              {model.n_neurons} neurons, Layer {j+1} of {model.n_layers}',
              fontsize=16)
    # setting the scale
    plt.xscale(scale)
    plt.yscale(scale)

    plt.xlabel('Rank', fontsize=16)
    plt.ylabel('eigenvalue', fontsize=16)
    plt.tick_params(axis='both', which='both', labelsize=16)

    #plt.xlim(min([min(spec_hist), min() 1]), len(spec_hist[0]))
    plt.ylim(np.max([np.min(spec_hist), 10**(-5)]), np.max(spec_hist))
    #plt.ylim(np.min(spec_hist), np.max([np.max(spec_hist), model.var]))



    plt.legend(reverse=True)

    if save_fig:
      plt.savefig(f'{image_saveloc}/spec_hist_{model.n_neurons}_{model.n_epochs}_layer{j+1}{saveadd}.png')
      #files.download(f'spec_hist_{model.n_neurons}_{model.n_epochs}_layer{j+1}.png')

    plt.show()
  return

def plot_spectrum_single(model, quantity, layer_ind, scale='log', save_fig=False):
    if quantity == 'activations':
        y_vals = model.activation_spectrum[layer_ind].detach().numpy()
    elif quantity == 'weights':
        y_vals = model.weight_spectrum[layer_ind].detach().numpy()
        y_init = model.spectrum_history[layer_ind][0].detach().numpy()

    if y_vals is None:
        raise Exception('Invalid quantity. Please select either "activations" or "weights"')


    x_vals = list(range(1, len(y_vals) + 1))

    fig = plt.plot(figsize=(10, 5))
    plt.plot(x_vals, y_vals)
    if quantity =='weights':
        plt.plot(x_vals, y_init, label='init')
        plt.legend()
    plt.title(f'{quantity} spectrum, layer {layer_ind + 1}', fontsize=16)
    plt.xscale(scale)
    plt.yscale(scale)

    plt.xlabel('Rank', fontsize=16)
    plt.ylabel('eigenvalue', fontsize=16)
    plt.tick_params(axis='both', which='both', labelsize=16)
    plt.show()

    return

def plot_spectrum_normed(model, scale='log', save_fig = True, xmax = 1000, saveadd=''):
  '''
  also plots effective dimensionality
  '''

  begin = 0
  for j in range(len(model.normed_spectra)):
    epochs = model.epoch_history
    relatives = model.normed_spectra[j]
    inits = model.adj_init_spectra[j]
    effdims = model.effective_dimensions[j]
    spec_hist = model.spectrum_history[j]

    colors = plt.cm.Spectral(np.linspace(0, 1, len(epochs) +1))
    #colors = plt.cm.viridis(np.linspace(0, 1, len(epochs) +1))

    fig = plt.figure(figsize=(10,5))


    # get the effective dimensionality points
    ed_ys = []

    for i in range(begin, len(spec_hist)):
      xs = list(range(1, len(model.spectrum_history[j][0]) + 1))
      ys = spec_hist[i]
      if i == 0:
          y2s = spec_hist[i]
      else:
          y2s = inits[i-1]



      # plotting the unrelative spectrum and the normalized init
      plt.plot(xs, ys, color=colors[i], label =f'epoch {epochs[i]}')
      plt.plot(xs, y2s, color=colors[i], linestyle=':', label=f'init norm {epochs[i]}')

      # get the intersection of the relative spectrum and the effective dimensionality
      #if i < len(epochs):
      y_val = np.interp(effdims[i], xs, ys)
      ed_ys.append(y_val)


    # plot the effective dimensions intersections
    #test_accuracies = [int(model.test_history[i]*100) for i in range(1,len(model.test_history))]
    test_accuracies = model.val_history[begin:]
    #acc_colors = plt.cm.Greys(np.linspace(0,1,100))
    plt.plot(effdims[begin:], ed_ys, 'k--', label = 'Eff. Dim.')

    # colormaps: spring, coolwarm, Greys
    plt.scatter(effdims[begin:], ed_ys, c=test_accuracies, cmap = plt.cm.spring, 
                s=20, edgecolors= "black", label = 'Val Accuracy', zorder=10)
    plt.colorbar()

    # add the baseline variance
    #plt.hlines(model.var, 1, len(ys), colors='k', linestyles = "dashed", label = f"init var = {model.var}")

    plt.title(f'Rescaled Init and Unnormalized Spectra \
              Network with {model.n_neurons} neurons, Layer {j+1} of {model.n_layers}',
              fontsize=16)
    plt.xscale(scale)
    plt.yscale(scale)

    plt.xlabel('Rank', fontsize=16)
    plt.ylabel('eigenvalue', fontsize=16)
    plt.tick_params(axis='both', which='both', labelsize=16)


    if xmax:
      xmax = xmax
    else:
      xmax = len(model.spectrum_history[0][0])

    plt.xlim(1, xmax)
    plt.ylim(np.max([np.min(spec_hist), 10**(-5)]), np.max(spec_hist))
#    plt.ylim(np.min(spec_hist), np.max([np.max(spec_hist), model.var]))


    #plt.legend(reverse=True)

    if save_fig:
      plt.savefig(f'{image_saveloc}/ED_spec_hist_{model.n_neurons}_{model.n_epochs}_layer{j+1}{saveadd}.png')
      # files.download(f'ED_spec_hist_{model.n_neurons}_{model.n_epochs}_layer{j+1}.png')

    plt.show()


def plot_accuracy(model, save_fig=False, saveadd=''):
    # setting up the figure
    fig = plt.figure(figsize=(10, 5))

    xs = [float(k) for k in model.epoch_history]
    plt.plot(xs, model.val_history, label=f'val')
    plt.plot(xs, model.train_history, label=f'train')

    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("accuracy", fontsize=16)

    plt.xscale("log")
    plt.xlim(-0.1, xs[-1]*1.1)

    plt.title(f"{model.n_neurons} neurons, {model.n_layers} layers performance",
              fontsize=16)
    plt.tick_params(axis='both', which='both', labelsize=16)

    if save_fig:
      plt.savefig(f'{image_saveloc}/accuracy_{model.n_neurons}_{model.n_epochs}{saveadd}.png')

    plt.legend()
    plt.show()

    return



def plot_accuracy_compare(model_list, save_fig=False, saveadd=''):
    """
    model_list :  list(tuple(SpectralAnalysis, str))  model object and name
    """
    # setting up the figure
    fig = plt.figure(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_list) + 1))

    i = 0
    for mod in model_list:
        model = mod[0]
        name = mod[1]
        xs = [float(k) for k in model.epoch_history]
        plt.plot(xs, model.val_history, color=colors[i], label=f'{name} val')
        plt.plot(xs, model.train_history, color=colors[i], linestyle=":", label=f'{name} train')
        i += 1

    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("accuracy", fontsize=16)

    plt.xscale("log")
    plt.xlim(-0.1, xs[-1] * 1.1)

    plt.title(f"comparing model performance", fontsize=16)
    plt.tick_params(axis='both', which='both', labelsize=16)

    if save_fig:
        plt.savefig(f'{image_saveloc}/accuracy_{len(model_list)}_compare{saveadd}.png')

    plt.legend()
    plt.show()

    return