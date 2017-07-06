import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import google_finance_io
import datetime
from dateutil import parser
from keras import backend as K

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cnn_keras_regr as cs
from keras.models import load_model


def main():
    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']

    # todo: fix problem encountered in 'ewu,qqq' etf - ugurgudelek

    # 1.download data
    google_finance_io.download_data(stock_names, start_date=datetime.date(2000, 1, 3),
                                    end_date=datetime.date(2016, 12, 31), verbose=True)

    # 2.calculate metric for available stocks and save them into csv file
    normalize_and_calculate_metrics(stock_names)

    # 3.calculate labels for available stocks and save them into csv file
    calculate_labels(stock_names)

    # 4.cluster features for available stocks and their features then save them into csv file
    cluster_features(stock_names, drop_this_cols=['date', 'low', 'close', 'high', 'open','adjusted_close'])

    # 5.read sorted (via hierarchical clustering) feature names from file
    sorted_cluster_names = pd.read_csv("../input/clustered_names.csv", header=None, squeeze=True).values.tolist()

    # 6. create flatten images with data and labels.
    create_images_from_data(stock_names, sorted_cluster_names, label_names=['label_day_tanh_regr'])

    # 7. merge all available data
    # data has 'images' and 'labels'
    data = get_merged_images_and_labels_data(stock_names, labels_are_last=1, train_test_ratio=0.9)

    # 8. call CNN
    params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}
    with K.get_session():
        cs.start_cnn_session(data, params,model_name="model_regr_100epoch")
    #
    # # draw some sample
    # # draw_image(data['images'].iloc[0].values.reshape(28, 28), sorted_cluster_names)
    # # plt.show()







    # plt.plot(sma_15_data, color='r')
    # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # plt.show()


if __name__ == "__main__":
    main()

    # data = get_last_saved_data()
    # model = load_model("../model/model_regr_100epoch_before_2017_06_16 21_55_06_953896")
    # params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}
    # with K.get_session():
    #     _, history = cs.test(model, data, params, q_ratio=0.8)
    #
    #     pd.to_pickle(history, "../result/history.pickle")

    # stock_names = ['spy', 'xlf', 'xlu', 'xle',
    #                'xlp', 'xli', 'xlv', 'xlk', 'ewj',
    #                'xlb', 'xly', 'eww', 'dia', 'ewg',
    #                'ewh', 'ewc', 'ewa']
    # cluster_features(stock_names, drop_this_cols=['date', 'low', 'close', 'high', 'open', 'adjusted_close'], hierarcy_no_plot=False)
    # plt.show()

