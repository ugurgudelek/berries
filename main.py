import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import metrics as mt
import yahoo_finance_io
import datetime
from dateutil import parser
import clustering
import classes
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cnn_handler as ch


def crop_data(arr):
    """crop the non-adjusted data starting from the beginning"""
    min_len = len(arr[0])
    for data in arr[1:]:
        min_len = min(min_len, len(data))

    for i in range(len(arr)):
        # crop beginning of data
        arr[i] = arr[i][len(arr[i]) - min_len:]

    return arr


def assign_null_into_data(arr, length):
    for i, data in enumerate(arr):
        null_len = length - len(data)
        null_part = np.zeros(null_len)
        null_part.fill(np.nan)

        arr[i] = np.hstack((null_part, data))
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


def calculate_metrics(data):

    metric_function_data = []

    rsi_15 = mt.rsi(data, 15); metric_function_data.append(rsi_15)
    rsi_20 = mt.rsi(data, 20); metric_function_data.append(rsi_20)
    rsi_25 = mt.rsi(data, 25); metric_function_data.append(rsi_25)
    rsi_30 = mt.rsi(data, 30); metric_function_data.append(rsi_30)
    # rsi_35 = mt.rsi(data, 35); metric_function_data.append(rsi_35)
    # rsi_40 = mt.rsi(data, 40); metric_function_data.append(rsi_40)
    # rsi_45 = mt.rsi(data, 45); metric_function_data.append(rsi_45)
#    rsi_50 = mt.rsi(data, 50); metric_function_data.append(rsi_50)

    sma_15 = mt.sma(data, 15); metric_function_data.append(sma_15)
    sma_20 = mt.sma(data, 20); metric_function_data.append(sma_20)
    sma_25 = mt.sma(data, 25); metric_function_data.append(sma_25)
    sma_30 = mt.sma(data, 30); metric_function_data.append(sma_30)
    # sma_35 = mt.sma(data, 35); metric_function_data.append(sma_35)
    # sma_40 = mt.sma(data, 40); metric_function_data.append(sma_40)
    # sma_45 = mt.sma(data, 45); metric_function_data.append(sma_45)
#    sma_50 = mt.sma(data, 50); metric_function_data.append(sma_50)

    macd_26_12 = mt.macd(data, 26, 12); metric_function_data.append(macd_26_12)
    macd_28_14 = mt.macd(data, 28, 14); metric_function_data.append(macd_28_14)
    macd_30_16 = mt.macd(data, 30, 16); metric_function_data.append(macd_30_16)
    # macd_32_18 = mt.macd(data, 32, 18); metric_function_data.append(macd_32_18)
    # macd_32_20 = mt.macd(data, 32, 20); metric_function_data.append(macd_32_20)
    # macd_34_22 = mt.macd(data, 34, 22); metric_function_data.append(macd_34_22)
    # macd_36_24 = mt.macd(data, 36, 24); metric_function_data.append(macd_36_24)
#    macd_38_26 = mt.macd(data, 38, 26); metric_function_data.append(macd_38_26)

    macd_trigger_9_26_12  = mt.macd_trigger(data, 9, 26, 12) ; metric_function_data.append(macd_trigger_9_26_12 )
    macd_trigger_10_28_14 = mt.macd_trigger(data, 10, 28, 14); metric_function_data.append(macd_trigger_10_28_14)
    macd_trigger_11_30_16 = mt.macd_trigger(data, 11, 30, 16); metric_function_data.append(macd_trigger_11_30_16)
    # macd_trigger_12_32_18 = mt.macd_trigger(data, 12, 32, 18); metric_function_data.append(macd_trigger_12_32_18)
    # macd_trigger_13_34_20 = mt.macd_trigger(data, 13, 34, 20); metric_function_data.append(macd_trigger_13_34_20)
    # macd_trigger_14_36_22 = mt.macd_trigger(data, 14, 36, 22); metric_function_data.append(macd_trigger_14_36_22)
    # macd_trigger_15_38_24 = mt.macd_trigger(data, 15, 38, 24); metric_function_data.append(macd_trigger_15_38_24)
#    macd_trigger_16_40_26 = mt.macd_trigger(data, 16, 40, 26); metric_function_data.append(macd_trigger_16_40_26)

    willR_14 = mt.williamsR(data, 14); metric_function_data.append(willR_14)
    willR_18 = mt.williamsR(data, 18); metric_function_data.append(willR_18)
    willR_22 = mt.williamsR(data, 22); metric_function_data.append(willR_22)
    # willR_26 = mt.williamsR(data, 26); metric_function_data.append(willR_26)
    # willR_30 = mt.williamsR(data, 30); metric_function_data.append(willR_30)
    # willR_34 = mt.williamsR(data, 34); metric_function_data.append(willR_34)
    # willR_38 = mt.williamsR(data, 38); metric_function_data.append(willR_38)
#    willR_42 = mt.williamsR(data, 42); metric_function_data.append(willR_42)

    kdHist_14 = mt.kdDiff(data, 14); metric_function_data.append(kdHist_14)
    kdHist_18 = mt.kdDiff(data, 18); metric_function_data.append(kdHist_18)
    kdHist_22 = mt.kdDiff(data, 22); metric_function_data.append(kdHist_22)
    # kdHist_26 = mt.kdDiff(data, 26); metric_function_data.append(kdHist_26)
    # kdHist_30 = mt.kdDiff(data, 30); metric_function_data.append(kdHist_30)
    # kdHist_34 = mt.kdDiff(data, 34); metric_function_data.append(kdHist_34)
    # kdHist_38 = mt.kdDiff(data, 38); metric_function_data.append(kdHist_38)
#    kdHist_42 = mt.kdDiff(data, 42); metric_function_data.append(kdHist_42)

    ultimateOs_7_14_28 = mt.ulOs(data, 7, 14, 28); metric_function_data.append(ultimateOs_7_14_28 )
    ultimateOs_8_16_32 = mt.ulOs(data, 8, 16, 32); metric_function_data.append(ultimateOs_8_16_32 )
    ultimateOs_9_18_36 = mt.ulOs(data, 9, 18, 36); metric_function_data.append(ultimateOs_9_18_36 )
    # ultimateOs_10_20_40 = mt.ulOs(data, 10, 20, 40); metric_function_data.append(ultimateOs_10_20_40)
    # ultimateOs_11_22_44 = mt.ulOs(data, 11, 22, 44); metric_function_data.append(ultimateOs_11_22_44)
    # ultimateOs_12_24_48 = mt.ulOs(data, 12, 24, 48); metric_function_data.append(ultimateOs_12_24_48)
    # ultimateOs_13_26_52 = mt.ulOs(data, 13, 26, 52); metric_function_data.append(ultimateOs_13_26_52)
#    ultimateOs_14_28_56 = mt.ulOs(data, 14, 28, 56); metric_function_data.append(ultimateOs_14_28_56)

    mfIndex_14 = mt.mfi(data, 14); metric_function_data.append(mfIndex_14)
    mfIndex_18 = mt.mfi(data, 18); metric_function_data.append(mfIndex_18)
    mfIndex_22 = mt.mfi(data, 22); metric_function_data.append(mfIndex_22)
    # mfIndex_26 = mt.mfi(data, 26); metric_function_data.append(mfIndex_26)
    # mfIndex_30 = mt.mfi(data, 30); metric_function_data.append(mfIndex_30)
#    mfIndex_34 = mt.mfi(data, 34); metric_function_data.append(mfIndex_34)
#    mfIndex_38 = mt.mfi(data, 38); metric_function_data.append(mfIndex_38)
#    mfIndex_40 = mt.mfi(data, 40); metric_function_data.append(mfIndex_40)

    # metric_function_names = ["rsi_15","rsi_20","rsi_25","rsi_30","rsi_35","rsi_40","rsi_45","rsi_50",
    #                         "sma_15","sma_20","sma_25","sma_30","sma_35","sma_40","sma_45","sma_50",
    #                         "macd_26_12","macd_28_14","macd_30_16","macd_32_18","macd_32_20","macd_34_22","macd_36_24","macd_38_26",
    #                         "macd_trigger_9_26_12","macd_trigger_10_28_14","macd_trigger_11_30_16","macd_trigger_12_32_18","macd_trigger_13_34_20",
    #                          "macd_trigger_14_36_22","macd_trigger_15_38_24","macd_trigger_16_40_26",
    #                         "willR_14","willR_18","willR_22","willR_26","willR_30","willR_34","willR_38","willR_42",
    #                         "kdHist_14","kdHist_18","kdHist_22","kdHist_26","kdHist_30","kdHist_34","kdHist_38","kdHist_42",
    #                         "ultimateOs_7_14_28","ultimateOs_8_16_32","ultimateOs_9_18_36","ultimateOs_10_20_40","ultimateOs_11_22_44","ultimateOs_12_24_48",
    #                          "ultimateOs_13_26_52","ultimateOs_14_28_56", "mfIndex_14","mfIndex_18","mfIndex_22","mfIndex_26","mfIndex_30","mfIndex_34","mfIndex_38","mfIndex_40"]

    # metric_function_names = ["rsi_15","rsi_20","rsi_25","rsi_30","rsi_35","rsi_40","rsi_45",
    #                         "sma_15","sma_20","sma_25","sma_30","sma_35","sma_40","sma_45",
    #                         "macd_26_12","macd_28_14","macd_30_16","macd_32_18","macd_32_20","macd_34_22","macd_36_24",
    #                         "macd_trigger_9_26_12","macd_trigger_10_28_14","macd_trigger_11_30_16","macd_trigger_12_32_18","macd_trigger_13_34_20",
    #                          "macd_trigger_14_36_22","macd_trigger_15_38_24",
    #                         "willR_14","willR_18","willR_22","willR_26","willR_30","willR_34","willR_38",
    #                         "kdHist_14","kdHist_18","kdHist_22","kdHist_26","kdHist_30","kdHist_34","kdHist_38",
    #                         "ultimateOs_7_14_28","ultimateOs_8_16_32","ultimateOs_9_18_36","ultimateOs_10_20_40","ultimateOs_11_22_44","ultimateOs_12_24_48",
    #                          "ultimateOs_13_26_52", "mfIndex_14","mfIndex_18","mfIndex_22","mfIndex_26","mfIndex_30"]
    metric_function_names = ["rsi_15","rsi_20","rsi_25","rsi_30",
                            "sma_15","sma_20","sma_25","sma_30",
                            "macd_26_12","macd_28_14","macd_30_16",
                            "macd_trigger_9_26_12","macd_trigger_10_28_14","macd_trigger_11_30_16",
                            "willR_14","willR_18","willR_22",
                            "kdHist_14","kdHist_18","kdHist_22",
                            "ultimateOs_7_14_28","ultimateOs_8_16_32","ultimateOs_9_18_36",
                             "mfIndex_14","mfIndex_18","mfIndex_22"]



    return np.asarray(metric_function_data),metric_function_names


def stack_data_and_metrics(data, metric_data, metric_function_names):
    for i, datum in enumerate(metric_data):
        data[metric_function_names[i]] = datum
    return data


def data_handler(which_stock, start_date, end_date, period = 28, is_save_csv=True):
    """Downloads data, calculates metrics and save results to separate .csv files for each ETF."""
    # Open High Low Close Volume Adj Close
    stock = yahoo_finance_io.data_getter(which_stock, start_date, end_date)
    if stock is not None:
        # stock = stock.as_matrix()[:, :][::-1]
        stock = stock.iloc[::-1]
        # Tranform array to dict for easy use
        # stock = tranform_to_dict(stock)

        # Adjust data according to adjusted close
        stock = adjusted_data(stock)

        # create data arr to hold all metric info
        metric_data, metric_function_names = calculate_metrics(stock)

        # assign nan value beginning of the data
        metric_data = assign_null_into_data(arr=metric_data, length=len(stock['adjusted_close']))

        # append data and metrics column-wise
        stock = stack_data_and_metrics(stock, metric_data, metric_function_names)

        # assign df targets
        target_df = classes.df_classes(stock['adjusted_close'].values, period=period, diff_thr=0.5)
        stock['label_df_is_less'] = target_df[:, 0]
        stock['label_df_is_same'] = target_df[:, 1]
        stock['label_df_is_more'] = target_df[:, 2]

        # assign lr targets
        target_lr = classes.lr_classes(stock['adjusted_close'].values, period=period, slope_quant=2)
        stock['label_lr_is_less'] = target_lr[:, 0]
        stock['label_lr_is_more'] = target_lr[:, 1]

        if is_save_csv:
            stock.to_csv("data/" + which_stock + ".csv", index=None)

        return stock

def cluster_features(p_stock_names):
    """Calls the clustering function after calculation of the metrics for all the ETFs.
    This function should be called after the metrics are calculated and saved to .csv files
    but before the images are constructed."""

    # read the first csv
    raw_data = pd.read_csv("data/{}.csv".format(p_stock_names[0]))
        
    # drop irrelevant features
    data = raw_data.drop(['date', 'low', 'close', 'high', 'open'], axis=1)
        
    # get predictor names for dropping processes
    predictor_names = [name for name in data.columns.values.tolist() if "label" not in name]
    data = data.dropna(subset=predictor_names)  # drop nan values for proper set

    # all data will be appended to this dataframe
    all_data = data
    
    for stock in p_stock_names[1:len(p_stock_names)]:
        
        raw_data = pd.read_csv("data/{}.csv".format(stock))
        
        # drop irrelevant features
        data = raw_data.drop(['date', 'low', 'close', 'high', 'open'], axis=1)
        
        # get predictor names for dropping processes
        predictor_names = [name for name in data.columns.values.tolist() if "label" not in name]
        data = data.dropna(subset=predictor_names)  # drop nan values for proper set

        all_data = all_data.append(data)

    # now, cluster features of the whole data
    sorted_predictor_names = clustering.hierarchical_clustering(all_data[predictor_names], no_plot=True)

    # save the names of the clustered features to file
    pd.Series(sorted_predictor_names).to_csv("clustered_names.csv", header=False, index=False)

    # return the names of the clustered features
    return sorted_predictor_names


def get_data(which_stock, p_sorted_predictor_names, split_period=28, label_names=['label_df_is_less', 'label_df_is_same', 'label_df_is_more', 'label_lr_is_less','label_lr_is_more']):
    """
    Reads metric data, clusters features, prepares and returns images together with labels.
    Images are flattened before returned.

    :param :type str which_stock: takes stock names

    :param :type int split_period=28 determines chuck size wrt date
    :param :type list label_names select label to use later for prediction
    :param :type bool cluster true: call clustering or false: use csv file

    :returns (list)images, (list)labels, (tuple)(image_row_size,image_col_size)
    """
    print("get_data called for {}".format(which_stock))
    raw_data = pd.read_csv("data/{}.csv".format(which_stock))

    # drop irrelevant features
    data = raw_data.drop(['date', 'low', 'close', 'high', 'open'], axis=1)

    # get predictor names for dropping processes
    # predictor_names = [name for name in data.columns.values.tolist() if "label" not in name]
    predictor_names = p_sorted_predictor_names
    data = data.dropna(subset=predictor_names)  # drop nan values for proper set

    image_col_size = data[predictor_names].shape[1]
    image_row_size = split_period

    # todo handle below 2 line - ugurgudelek
    # if image_row_size != image_col_size:
    #     raise Exception("image matrix must be square!")

    data = data[predictor_names + label_names]

    images = []
    labels = []
    merged = []
    # split image chunks
    for i in range(split_period - 1, data.shape[0] - split_period):
        lower = i - split_period + 1
        upper = lower + split_period

        image = data[predictor_names].iloc[lower:upper]

        # change raw data to regular gray scale image
        image = image.apply(lambda x: (((x - x.min()) / (x.max() - x.min())) * 255).round(), axis=0)

        image_flat = image.values.flatten()  # image_flat'shape : image_row_size * image_col_size
        label = data[label_names].iloc[upper - 1].values

        images.append(image_flat)
        labels.append(label)

    return images, labels, (image_row_size, image_col_size)

def prepare_images(prepare_data=True, save_etf=False, is_cluster_features=False):
    """Stock names are determined and funtions that calculate metrics and prepare images are called.
    Images are saved to separate .csv files for each ETF"""

    # stock_names = ['spy', 'gdx', 'xlf', 'jnug', 'eem', 'nugt', 'vxx', 'iwm', 'gdxj', 'uso', 'efa', 'uvxy', 'qqq', 'fxi',
    #                'jdst', 'ewz', 'xlu', 'xle', 'ung', 'xiv', 'xop', 'vwo', 'xlp', 'hyg', 'jnk', 'xli', 'tlt', 'tza',
    #                'xlv', 'rsx', 'ugaz', 'amlp', 'dust', 'vea', 'iemg', 'uco', 'xlk', 'iau', 'gld', 'kre', 'sds', 'iyr',
    #                'xrt', 'slv', 'dgaz', 'ewj', 'xbi', 'oih', 'ezu', 'xlb', 'lqd', 'bkln', 'vnq', 'ijh', 'labd', 'xly',
    #                'iefa', 'spxu', 'xme', 'ewt', 'dxj', 'eww', 'spxs', 'dia', 'fas', 'ivv', 'ijr', 'tna', 'inda', 'ewg',
    #                'pff', 'vgk', 'svxy', 'agg', 'ewh', 'dbc', 'ewc', 'iwf', 'epi', 'kbe', 'sso', 'vixy', 'oil', 'labu',
    #                'tbt', 'sqqq', 'itb', 'tqqq', 'ewu', 'bnd', 'ewa', 'vti', 'voo', 'fez', 'emb', 'iwd', 'uup', 'ewy',
    #                'fxn', 'xlre']

    if prepare_data:
        
        # stock_names = ['spy', 'xlf', 'qqq', 'xlu' , 'xle' , 'xlp' , 'xli' , 'xlv' , 'xlk' , 'ewj' , 'xlb', 'xly', 'eww',
          #              'dia', 'ewg', 'ewh', 'ewc', 'ewu','ewa']
        stock_names = ['spy', 'xlf']

        # CALCULATE METRICS, CREATE DATASET CSVs
        # example of yahoo finance data getter function
        start_date = datetime.date(2000, 1, 3)
        end_date = datetime.date(2017, 1, 1)

        # get available ETFs (some ETFs lack sufficient data)
        gaugeCounter = 1
        available_etfs = []
        for stock in stock_names:
            print(gaugeCounter)
            df = data_handler(stock, start_date, end_date, is_save_csv=True)
            if df is not None:
                available_etfs.append(stock)
                print(stock)
            gaugeCounter += 1

        # save available etfs for later use
        if save_etf:
            pd.DataFrame(available_etfs).to_csv("available_etfs.csv", header=False, index=False)

        if is_cluster_features == True:    
            sorted_predictor_names = cluster_features(stock_names)
        else:
            sorted_predictor_names = pd.read_csv("clustered_names.csv", header=None, squeeze=True).values.tolist()

    # get all flatten images and labels for cnn
    # read available etfs
    available_etfs = pd.read_csv("available_etfs.csv", header=None, squeeze=True).values.tolist()
    for etf in available_etfs:
        images, labels, (image_row_size, image_col_size) = get_data(etf, p_sorted_predictor_names = sorted_predictor_names)

        data_df = pd.concat([pd.DataFrame(images), pd.DataFrame(labels)],axis=1)

        data_df.to_csv("images/{}_images_labels.csv".format(etf), index=False)

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
    train_labels = data['labels'].iloc[:train_size, -2:] #select last 2 column as a label

    test_images = data['images'].iloc[train_size:]
    test_labels = data['labels'].iloc[train_size:, -2:]

    return train_images,train_labels,test_images,test_labels

def draw_image(image,xtick_labels, cmap=None):
    # draw image
    fig = plt.figure()
    plt.xticks(list(range(len(xtick_labels))), xtick_labels, rotation='vertical')
    plt.imshow(image, cmap=cmap)


def main():

    # prepare_images(prepare_data=True, save_etf=False, is_cluster_features=False)
    
    # # read available etfs
    # available_etfs = pd.read_csv("available_etfs.csv", header=None, squeeze=True).values.tolist()
    
    # # available_etfs = ['spy']
    # # READ IMAGES DIRECTLY
    # all_images = []
    # all_labels = []
    # for etf in available_etfs:
    #     data_df = pd.read_csv("images/{}_images_labels.csv".format(etf))
    #     images = data_df.iloc[:,:-5]
    #     labels = data_df.iloc[:,-5:]
    
    #     print("images are merging with {}".format(etf))
    #     if len(all_images) == 0:
    #         all_images = np.array(images)
    #         all_labels = labels.values
    #     else:
    #         all_images = np.append(all_images, images, axis=0)
    #         all_labels = np.append(all_labels, labels.values, axis=0)
    
    #     print(pd.DataFrame(all_images).shape)
    
    
    # data = {'images':pd.DataFrame(all_images), 'labels':pd.DataFrame(all_labels)}
    
    # # save to pickle
    # pd.to_pickle(data, "data.pickle")

    # read from pickle
    data = pd.read_pickle("data.pickle")

    # plot some samples
    # sorted_cluster_names = pd.read_csv("clustered_names.csv", header=None, squeeze=True).values.tolist()
    # draw_image(data['images'].iloc[0].values.reshape(28,28), sorted_cluster_names)
    # draw_image(data['images'].iloc[5000].values.reshape(28, 28), sorted_cluster_names)
    # draw_image(data['images'].iloc[10000].values.reshape(28, 28), sorted_cluster_names)
    # draw_image(data['images'].iloc[20000].values.reshape(28, 28), sorted_cluster_names)
    # draw_image(data['images'].iloc[30000].values.reshape(28, 28), sorted_cluster_names)
    # plt.show()

    # shuffle data
    data = shuffle_data(data)

    #train test split
    train_size = int(data['images'].shape[0]*0.95)
    train_images, train_labels, test_images, test_labels = train_test_split(data, train_size=train_size)


    # call CNN
    parameters = {'learning_rate': 0.001, 'training_iters': train_size, 'batch_size': 64, 'dropout': 0.6}
    ch.launch_cnn(train_images,train_labels,test_images,test_labels, image_shape=(28,28), parameters=parameters)

    # plt.plot(sma_15_data, color='r')
    # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # plt.show()


if __name__ == "__main__":
    main()
