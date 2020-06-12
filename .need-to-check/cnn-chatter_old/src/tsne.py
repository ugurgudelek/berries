__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torchvision.datasets import MNIST

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio


import torch

from sklearn.datasets import make_blobs

from dataset import TimeSeriesDataset, EnergyRADataset


def cluster_plot(X, y, hover, model, filename=None, backend='plotly', auto_open=False):
    """

    Args:
        X (np.ndarray): 2 dimensional feature array -> size(batch_size, channel*height*widht)
        y (np.ndarray): 1 dimensional corresponding label array -> size(batch_size)
        backend: type of plotting library. Available options are 'plotly' and 'seaborn'

    """

    reduced_data = model.fit_transform(X)

    reduced_df = pd.DataFrame(data={'X': reduced_data[:, 0],
                                    'Y': reduced_data[:, 1],
                                    'label': y,
                                    'hover':hover})
    if backend == 'plotly':
        data = [go.Scatter(x=group['X'],
                           y=group['Y'],
                           mode='markers',
                           name=label,
                           hoverinfo='text',
                           hovertext=group['hover'])
                for label, group in reduced_df.groupby('label')]

        layout = go.Layout(
            xaxis={'title': 'X'},
            yaxis={'title': 'Y'}
        )

        fig = go.Figure(data=data, layout=layout)

        pio.write_html(fig, filename)

        fig.show()


    if backend == 'seaborn':
        g = sns.FacetGrid(reduced_df, hue='label', height=6).map(plt.scatter, 'X', 'Y')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # SUBSET_SIZE = 1000
    # # dataset = MNIST('../data/MNIST', transform=img_transform, download=True)
    # # X = dataset.train_data[:SUBSET_SIZE].view(-1, 28 * 28).numpy()
    # # y = dataset.train_labels.numpy()[:SUBSET_SIZE].astype(int)
    #
    dataset = TimeSeriesDataset().dataset
    # dataset = EnergyRADataset(label_col_name='Sec_cut (j/mm3)').dataset
    X, y = dataset.features, dataset.labels

    y = np.vectorize({  0:'no chatter',
                        1:'medium chatter',
                        2:'high chatter'}.get)(y)
    cluster_plot(X, y, hover=dataset.dataframe['slotname'].values,
                 model=TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200, n_iter=2000),
              backend='seaborn', filename='EnergyRADataset.html')
    #
    #
    # # cluster_plot(X, y, model=KMeans(n_clusters=10, random_state=0),
    # #           backend='seaborn')

