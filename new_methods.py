import pandas as pd
import numpy as np
import os




def split(series, threshold):
    """
    Splits given series into 3 sets wrt threshold
    :param (pd.Series) series: 
    :param (float) threshold: 
    :return: (pd.series,pd.series,pd.series) 3 different series
    
    
    """
    return (series.loc[series < -threshold],
            series.loc[np.logical_and(series <= threshold,series >= -threshold)],
            series.loc[series > threshold])


def split_wrt_min_var(series):
    """
    Finds a value while minimizing splitted chunk variance
    :param (pd.Series) series: 
    :return: (tuple(float,float)) (np.argmin, np.min)
    """
    var_list = []
    for i in range(100):
        ratio = i/100
        below,same,above = split(series, ratio)
        var_list.append(below.var() + same.var() + above.var())
    return np.argmin(var_list) / 100, np.min(var_list)  # argmin of variance list


def apply_normalization_to_raw_data(stock):

    # insert percent change column into main dataframe
    percentage = stock.adjusted_close.pct_change(periods=1)
    percentage.iloc[0] = 0.0
    percentage_100 = percentage * 100
    stock['pct_change_tanh'] = percentage_100.apply(np.tanh)
    return stock