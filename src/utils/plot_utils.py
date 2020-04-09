# -*- coding: utf-8 -*-
# @Time   : 3/23/2020 11:45 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          save_path=None,
                          title='Confusion matrix',
                          cmap=plt.get_cmap('Blues')
                          ):


    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure()

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n{cm_norm[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Commented section fixes a bug on some matplotlib version
    # ylim = ax.get_ylim()
    # ax.set_ylim(ylim[0] + 0.5, ylim[1] - 0.5)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    return ax

def image_folder_to_gif(fpath, glob=None):
    import imageio
    from pathlib import Path
    from pygifsicle import optimize
    from tqdm import tqdm
    fpath = Path(fpath)
    filenames = list(fpath.glob(glob or '*.jpg'))
    with imageio.get_writer(fpath/'animated.gif', mode='I') as writer:
        with tqdm(total=len(filenames), desc='Reading images..') as pbar:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                pbar.update(1)
    optimize(source=str(fpath)+'/animated.gif',
             destination=str(fpath)+'/optimized.gif',
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