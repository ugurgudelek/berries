import numpy as np
from scipy import stats
import pandas as pd


def lr_classes(prices, period=28, slope_quant=2):
    """prices: adjusted price values of the asset.
    period: number of values for which lr will be calculated.
    slope_quant: quantization of the slope (how many classes).
    Return value is one-hot encoded.
    Classes are assigned to instances starting from the index (period - 1) until
    (prices.shape - period - 1) both ends inclusive.
    E.g. class of the first instance is calculated using the data from 
    (period) to (2 * period - 1)."""

    classes = np.zeros((prices.shape[0], slope_quant))

    for i in range(period - 1, prices.shape[0] - period):

        curRangePrices = prices[i + 1: i + 1 + period]
        ranges = np.linspace(1, curRangePrices.shape[0], curRangePrices.shape[0], endpoint=True)
        slope, _, _, _, _ = stats.linregress(ranges, curRangePrices)
        angle = np.arctan(slope) * 180 / np.pi
        quantranges = np.linspace(-90, 90, slope_quant + 1, endpoint=True)

        for j in range(1, quantranges.shape[0]):
            if (angle >= quantranges[j - 1] and angle < quantranges[j]):
                classes[i, j - 1] = 1
                break
    # classes[0:period, :] = np.nan
    # classes[prices.shape[0] - period-1:, :] = np.nan
    return classes


def df_classes(prices, period=28, diff_thr=0.5):
    """prices: adjusted price values of the asset.
    period: range for which the difference between the start and end values will be calculated
    diff_thr: threshold until which difference will be omitted (assumed to be the same).
    There will be 3 classes: less, same and more.
    Return value is one-hot encoded.
    Class assignment is the same as the function lr_classes."""

    classes = np.zeros((prices.shape[0], 3))

    for i in range(period - 1, prices.shape[0] - period):

        curDiff = prices[i + period] - prices[i+1]

        if abs(curDiff) < diff_thr:
            classes[i, 1] = 1
        elif curDiff < 0:
            classes[i, 0] = 1
        elif curDiff > 0:
            classes[i, 2] = 1

    # classes[0:period,:] = np.nan
    # classes[prices.shape[0] - period-1:,:] = np.nan
    return classes

def day_by_day_classes(prices, period=28):
    """
    :param prices: adjusted price values of the asset.
    :param period: number of values for which lr will be calculated.
    :return: one hot encoded classes
    
    returns classes which is one day after of training
    classes[:,0] := less
    classes[:,1] := more
    """

    classes = np.zeros((prices.shape[0], 2))

    for i in range(period - 1, prices.shape[0] - 1):

        diff = prices[i + 1] - prices[i]

        if diff <= 0:
            classes[i,0] = 1
        else:
            classes[i,1] = 1

    return classes

def day_by_day_reg(changes, period=28):

    """
    
    :param (pd.Series) changes: 
    :param (int) period: 
    :return: (pd.Series)
    """

    classes = changes.shift(periods=-1)
    classes.iloc[0:period - 1] = np.nan
    return classes







