import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from checked_methods import *
from new_methods import *
import google_finance_io
import datetime
from dateutil import parser

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import cnn_handler as ch
import cnn_keras_day as cs


def crop_data(arr):
    """crop the non-adjusted data starting from the beginning"""
    min_len = len(arr[0])
    for data in arr[1:]:
        min_len = min(min_len, len(data))

    for i in range(len(arr)):
        # crop beginning of data
        arr[i] = arr[i][len(arr[i]) - min_len:]

    return arr


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


def main():
    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']

    # todo: fix problem encountered in 'ewu,qqq' etf - ugurgudelek

    # 1.download data
    google_finance_io.download_data(stock_names, start_date=datetime.date(2000, 1, 3),
                                    end_date=datetime.date(2016, 12, 31), verbose=True)

    # todo : metric diversity sini arttır.

    # 2.calculate metric for available stocks and save them into csv file
    calculate_metrics_for_raw_data(stock_names)

    # done: normalize (in calculate_labels)

    # 3.calculate labels for available stocks and save them into csv file
    calculate_labels(stock_names)



    # 4.cluster features for available stocks and their features then save them into csv file
    cluster_features(stock_names)

    # todo: clustering olmadan ne oluyor?

    # 5.read sorted (via hierarchical clustering) feature names from file
    sorted_cluster_names = pd.read_csv("clustered_names.csv", header=None, squeeze=True).values.tolist()

    # 6. create flatten images with data and labels.
    create_images_from_data(stock_names, sorted_cluster_names)


    # 7. merge all available data
    # data has 'images' and 'labels'
    data = get_merged_images_and_labels_data(stock_names, labels_are_last=2, train_test_ratio=0.9)

    # 8. call CNN
    params = {"input_w": 28, "input_h": 28, "num_classes": 2, "batch_size": 1024, "epochs": 100}

    cs.train_cnn(data, params)

    # draw some sample
    # draw_image(data['images'].iloc[0].values.reshape(28, 28), sorted_cluster_names)
    # plt.show()

    # done: 1. veriyi artış haline getir.

    # todo: 2. veriden istatistik- bununla thresholdları ayarla

    # todo: 3. labelları 3 e çıkar.  # todo: biraz fuzzy logic gibi regression

    # todo: 4. kar-zarar hesaplamak lazım.

    # todo: 5. sürekli artış hipotezi. - buy-hold karşılaştırması.

    # todo: 6. resim

    # todo: 7. literature - related work - finance lstm,rnn

    # todo: 8. hisselere ve yıla göre başarım

    # todo: 9. hisse senetlerinin artırmak lazım.





    # plt.plot(sma_15_data, color='r')
    # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # plt.show()


if __name__ == "__main__":
    main()
    # check_for_null("input/images_with_labels")