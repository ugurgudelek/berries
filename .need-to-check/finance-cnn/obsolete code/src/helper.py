import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def quantize(x, ratio=0.38):
    if x > ratio:
        return 1
    elif x < -ratio:
        return -1
    else:
        return 0

def assign_null_into_data(arr, length):
    for i, data in enumerate(arr):
        null_len = length - len(data)
        null_part = np.zeros(null_len)
        null_part.fill(np.nan)

        arr[i] = np.hstack((null_part, data))
    return arr


def stack_data_and_metrics(data, metric_data, metric_function_names):
    for i, datum in enumerate(metric_data):
        data[metric_function_names[i]] = datum
    return data


def split(series, threshold):
    """
    Splits given series into 3 sets wrt threshold
    :param (pd.Series) series:
    :param (float) threshold:
    :return: (pd.series,pd.series,pd.series) 3 different series


    """
    return (series.loc[series < -threshold],
            series.loc[np.logical_and(series <= threshold, series >= -threshold)],
            series.loc[series > threshold])


def split_wrt_min_var(series):
    """
    Finds a value while minimizing splitted chunk variance
    :param (pd.Series) series:
    :return: (tuple(float,float)) (np.argmin, np.min)
    """
    var_list = []
    for i in range(100):
        ratio = i / 100
        below, same, above = split(series, ratio)
        var_list.append(below.var() + same.var() + above.var())
    return np.argmin(var_list) / 100, np.min(var_list)  # argmin of variance list




def adjusted_data(data):
    # adjust the prices according to adjusted close
    adj_ratio = data['adjusted_close'] / data['close']
    data['open'] = adj_ratio * data['open']  # open
    data['high'] = adj_ratio * data['high']  # high
    data['low'] = adj_ratio * data['low']  # low
    data['close'] = data['adjusted_close']  # close is adjusted

    return data


# # tempopary easy to use calculate_metrics.
# # needs to be extended or replaced with below
# def calculate_metrics(data):
#     # lets calculate some metric over dataset
#     metric_function_data = []
#
#     rsi_15_data = mt.rsi(data);
#     metric_function_data.append(rsi_15_data)
#     sma_15_data = mt.sma(data);
#     metric_function_data.append(sma_15_data)
#     macd_15_5_data = mt.macd(data);
#     metric_function_data.append(macd_15_5_data)
#     macd_trigger_9_15_5 = mt.macd_trigger(data);
#     metric_function_data.append(macd_trigger_9_15_5)
#     willR = mt.williamsR(data);
#     metric_function_data.append(willR)
#     kdHist = mt.kdDiff(data);
#     metric_function_data.append(kdHist)
#     ultimateOs = mt.ulOs(data);
#     metric_function_data.append(ultimateOs)
#     mfIndex = mt.mfi(data);
#     metric_function_data.append(mfIndex)
#
#     metric_function_names = ["rsi_15", "sma_15", "macd_15_5", "macd_trigger_9_15_5", "willR", "kdHist", "ultimateOs",
#                              "mfIndex"]
#     return np.asarray(metric_function_data), metric_function_names

# stock_names = ['spy', 'gdx', 'xlf', 'jnug', 'eem', 'nugt', 'vxx', 'iwm', 'gdxj', 'uso', 'efa', 'uvxy', 'qqq', 'fxi',
#                'jdst', 'ewz', 'xlu', 'xle', 'ung', 'xiv', 'xop', 'vwo', 'xlp', 'hyg', 'jnk', 'xli', 'tlt', 'tza',
#                'xlv', 'rsx', 'ugaz', 'amlp', 'dust', 'vea', 'iemg', 'uco', 'xlk', 'iau', 'gld', 'kre', 'sds', 'iyr',
#                'xrt', 'slv', 'dgaz', 'ewj', 'xbi', 'oih', 'ezu', 'xlb', 'lqd', 'bkln', 'vnq', 'ijh', 'labd', 'xly',
#                'iefa', 'spxu', 'xme', 'ewt', 'dxj', 'eww', 'spxs', 'dia', 'fas', 'ivv', 'ijr', 'tna', 'inda', 'ewg',
#                'pff', 'vgk', 'svxy', 'agg', 'ewh', 'dbc', 'ewc', 'iwf', 'epi', 'kbe', 'sso', 'vixy', 'oil', 'labu',
#                'tbt', 'sqqq', 'itb', 'tqqq', 'ewu', 'bnd', 'ewa', 'vti', 'voo', 'fez', 'emb', 'iwd', 'uup', 'ewy',
#                'fxn', 'xlre']



def shuffle_data(data):
    sort_order = np.random.permutation(len(data['images']))
    data['images'] = data['images'].iloc[sort_order]
    data['labels'] = data['labels'].iloc[sort_order]
    return data


def train_test_split(data, train_size):
    # split test and train
    # train_size = int(data['images'].shape[0]*0.9)

    print("{} train images selected.".format(train_size))
    train_images = data['images'].iloc[:train_size]
    train_labels = data['labels'].iloc[:train_size, -2:]  # select last 2 column as a label

    test_images = data['images'].iloc[train_size:]
    test_labels = data['labels'].iloc[train_size:, -2:]

    return train_images, train_labels, test_images, test_labels


def draw_image(image, xtick_labels, cmap=None):
    # draw image
    fig = plt.figure()
    plt.xticks(list(range(len(xtick_labels))), xtick_labels, rotation='vertical')
    plt.imshow(image, cmap=cmap)

def check_for_null(path):
    for file in os.listdir(path):
        df = pd.read_csv(path+"/"+file)

        print("{} : {}".format(file,df.isnull().sum().sum()))
