# -*- coding: utf-8 -*-
# @Time   : 3/23/2020 11:45 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : plot.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plotly.graph_objs import *
import plotly
import io


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes,
                          save_path=None,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues'),
                          figsize=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()

    # Compute confusion matrix
    cm = confusion_matrix(y_true,
                          y_pred,
                          labels=[i for i in range(len(classes))])
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        # title=title,
        ylabel='True label',
        xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=45,
             ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,
                    i,
                    f'{cm[i, j]}\n{cm_norm[i, j]:.2f}',
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Commented section fixes a bug on some matplotlib version
    # ylim = ax.get_ylim()
    # ax.set_ylim(ylim[0] + 0.5, ylim[1] - 0.5)

    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)
    return ax


def image_folder_to_gif(fpath, glob=None):
    import imageio
    from pathlib import Path
    from pygifsicle import optimize
    from tqdm import tqdm
    fpath = Path(fpath)
    filenames = list(fpath.glob(glob or '*.jpg'))
    with imageio.get_writer(fpath / 'animated.gif', mode='I') as writer:
        with tqdm(total=len(filenames), desc='Reading images..') as pbar:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                pbar.update(1)
    optimize(source=str(fpath) + '/animated.gif',
             destination=str(fpath) + '/optimized.gif',
             colors=10,
             options=["--verbose"])


def camera_ready_matplotlib_style(plot_func):

    def wrapper(*args, **kwargs):
        ax = plot_func(*args, **kwargs)

        # Font size
        # rc('font', size=28)
        # rc('font', family='serif')
        # rc('axes', labelsize=32)

        [t.set_va('center') for t in ax.get_yticklabels()]
        [t.set_ha('left') for t in ax.get_yticklabels()]
        [t.set_va('center') for t in ax.get_xticklabels()]
        [t.set_ha('right') for t in ax.get_xticklabels()]
        [t.set_va('center') for t in ax.get_zticklabels()]
        [t.set_ha('left') for t in ax.get_zticklabels()]

        # Background
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Tick Placement
        # ax.xaxis._axinfo['tick']['inward_factor'] = 0
        # ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
        # ax.yaxis._axinfo['tick']['inward_factor'] = 0
        # ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
        # ax.zaxis._axinfo['tick']['inward_factor'] = 0
        # ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
        # ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

        # ax.xaxis.set_major_locator(MultipleLocator(5))
        # ax.yaxis.set_major_locator(MultipleLocator(5))
        # ax.zaxis.set_major_locator(MultipleLocator(0.01))
        return ax

    return wrapper


def mpl2pillow(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    # buf.close()
    plt.close(fig)
    return img


def plot_clustering(z_run,
                    labels,
                    engine='plotly',
                    title_postfix='z',
                    download=False,
                    folder_name='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """

    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        print(len(hex_colors))
        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12,
                          n_iter=3000).fit_transform(z_run)

        trace = Scatter(x=z_run_pca[:, 0],
                        y=z_run_pca[:, 1],
                        mode='markers',
                        marker=dict(color=colors))
        data = Data([trace])
        layout = Layout(title='PCA on z_run', showlegend=False)
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(x=z_run_tsne[:, 0],
                        y=z_run_tsne[:, 1],
                        mode='markers',
                        marker=dict(color=colors))
        data = Data([trace])
        layout = Layout(title='tSNE on z_run', showlegend=False)
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, title_postfix, download,
                                   folder_name):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = ['tab:blue', 'tab:orange', 'tab:green']
        patches = [
            mpatches.Patch(color='tab:blue', label='Class 0'),
            mpatches.Patch(color='tab:orange', label='Class 1'),
            mpatches.Patch(color='tab:green', label='Class 2')
        ]
        # for _ in np.unique(labels):
        #     hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12,
                          n_iter=3000).fit_transform(z_run)

        plt.scatter(z_run_pca[:, 0],
                    z_run_pca[:, 1],
                    c=colors,
                    marker='.',
                    linewidths=0)

        plt.legend(handles=patches)
        plt.title(f'PCA on {title_postfix}')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.makedirs(folder_name, exist_ok=True)
            plt.savefig(folder_name + "/pca.png")

        plt.show()

        plt.scatter(z_run_tsne[:, 0],
                    z_run_tsne[:, 1],
                    c=colors,
                    marker='.',
                    linewidths=0)
        plt.legend(handles=patches)
        plt.title(f'tSNE on {title_postfix}')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.makedirs(folder_name, exist_ok=True)
            plt.savefig(folder_name + "/tsne.png")

        plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, title_postfix, download,
                                   folder_name)


def plot_clustering_legacy(z_run,
                           labels,
                           engine='plotly',
                           download=False,
                           folder_name='clustering'):
    """
    Given latent variables for all timeseries, and output of k-means, run PCA and tSNE on latent vectors and color the points using cluster_labels.
    :param z_run: Latent vectors for all input tensors
    :param labels: Cluster labels for all input tensors
    :param engine: plotly/matplotlib
    :param download: If true, it will download plots in `folder_name`
    :param folder_name: Download folder to dump plots
    :return:
    """

    def plot_clustering_plotly(z_run, labels):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12,
                          n_iter=3000).fit_transform(z_run)

        trace = Scatter(x=z_run_pca[:, 0],
                        y=z_run_pca[:, 1],
                        mode='markers',
                        marker=dict(color=colors))
        data = Data([trace])
        layout = Layout(title='PCA on z_run', showlegend=False)
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

        trace = Scatter(x=z_run_tsne[:, 0],
                        y=z_run_tsne[:, 1],
                        mode='markers',
                        marker=dict(color=colors))
        data = Data([trace])
        layout = Layout(title='tSNE on z_run', showlegend=False)
        fig = Figure(data=data, layout=layout)
        plotly.offline.iplot(fig)

    def plot_clustering_matplotlib(z_run, labels, download, folder_name):

        labels = labels[:z_run.shape[0]]  # because of weird batch_size

        hex_colors = []
        for _ in np.unique(labels):
            hex_colors.append('#%06X' % randint(0, 0xFFFFFF))

        colors = [hex_colors[int(i)] for i in labels]

        z_run_pca = TruncatedSVD(n_components=3).fit_transform(z_run)
        z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12,
                          n_iter=3000).fit_transform(z_run)

        plt.scatter(z_run_pca[:, 0],
                    z_run_pca[:, 1],
                    c=colors,
                    marker='*',
                    linewidths=0)
        plt.title('PCA on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/pca.png")
        else:
            plt.show()

        plt.scatter(z_run_tsne[:, 0],
                    z_run_tsne[:, 1],
                    c=colors,
                    marker='*',
                    linewidths=0)
        plt.title('tSNE on z_run')
        if download:
            if os.path.exists(folder_name):
                pass
            else:
                os.mkdir(folder_name)
            plt.savefig(folder_name + "/tsne.png")
        else:
            plt.show()

    if (download == False) & (engine == 'plotly'):
        plot_clustering_plotly(z_run, labels)
    if (download) & (engine == 'plotly'):
        print("Can't download plotly plots")
    if engine == 'matplotlib':
        plot_clustering_matplotlib(z_run, labels, download, folder_name)