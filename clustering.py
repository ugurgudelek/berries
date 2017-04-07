from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hierarchical_clustering(features_df):
    # dist = np.array([662., 877., 255., 412., 996., 295., 468., 268., 400., 754., 564., 138., 219., 869., 669.])

    pearson_corr = features_df.corr(method='pearson')
    dist = 1 - np.array(pearson_corr)


    linkage = hierarchy.linkage(dist, method='average', metric='euclidean')
    plt.figure()
    dn = hierarchy.dendrogram(linkage, labels=features_df.columns.values, no_plot=True)
    return dn['ivl']