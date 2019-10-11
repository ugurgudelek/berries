from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hierarchical_clustering(features_df, no_plot=True):

    pearson_corr = features_df.corr(method='pearson')
    dist = 1 - np.array(pearson_corr)

    linkage = hierarchy.linkage(dist, method='average', metric='euclidean')
    plt.figure()
    dn = hierarchy.dendrogram(linkage, labels=features_df.columns.values, no_plot=no_plot)
    
    return dn['ivl']
