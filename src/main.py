import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import *
import loss_profit
import google_finance_io
import datetime
from dateutil import parser
import os

from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cnn_keras_regr as cs
import cnn_keras_class as cscls
#import mlp_keras_regr as ms
from keras.models import load_model

def main(regression = True):
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

    if regression == True:

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
            cs.start_cnn_session(data, params, model_save_name="model_regr_100epoch", model_read_name = "model_regr_100epoch_before_2017_07_07 17_52_36_962776")
            
        # for mlp
        #params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}
        #with K.get_session():
            #ms.start_mlp_session(data, params, model_save_name="model_mlp_regr_100epoch", model_read_name = "") 
        
        #
        # # draw some sample
        # # draw_image(data['images'].iloc[0].values.reshape(28, 28), sorted_cluster_names)
        # # plt.show()

        # plt.plot(sma_15_data, color='r')
        # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
        # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
        # plt.show()

    else:
        
        calculate_labels(stock_names)
        cluster_features(stock_names, drop_this_cols=['date', 'low', 'close', 'high', 'open','adjusted_close'])
        sorted_cluster_names = pd.read_csv("../input/clustered_names.csv", header=None, squeeze=True).values.tolist()
        create_images_from_data(stock_names, sorted_cluster_names, label_names=['label_day_is_less', 'label_day_is_more'])
        data = get_merged_images_and_labels_data_cls(stock_names, last_image_col = -3, labels_ind = [-2, -1], train_test_ratio = 0.9)
        params = {"input_w": 28, "input_h": 28, "num_classes": 2, "batch_size": 1024, "epochs": 100}
        with K.get_session():
            cscls.start_cnn_session(data, params, model_save_name="model_class_100epoch", model_read_name = "model_class_100epoch_before_2017_07_13 12_22_28_390356")


if __name__ == "__main__":
    
    #main(regression = True)

    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']

    print("Preparing adjusted close dataframe...")
    prices = loss_profit.prepare_adj_close(stock_names)

    print("Calculating final capital using prediction model...")
    #capital, shares,_,_ = loss_profit.buy_sell_regr(predictions_name = 'predictions_model_regr_100epoch_2017_07_11 16_24_27_177432', adj_close = prices, buy_thr=.38, sell_thr=-.38, transaction_cost=0)
    #capital, shares = loss_profit.buy_sell_class3(predictions_name = 'predictions_model_class_100epoch_2017_07_11 17_08_21_002137', adj_close = prices, transaction_cost=5)
    capital, shares = loss_profit.buy_sell_class2(predictions_name = 'predictions_model_class_100epoch_2017_07_13 13_15_09_200788', adj_close = prices, transaction_cost=5)
   
    print("Final captial:")
    print(capital)
    print("Final shares:")
    print(shares)
    
    print("-----------------------------------------")
    print("Calculating final capital using buy & hold...")
    
    buy_hold_final_capital, buy_hold_final_shares = loss_profit.buy_hold(stock_names, prices)
    print("Final captial:")
    print(buy_hold_final_capital)
    print("Final shares:")
    print(buy_hold_final_shares)
    
    
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

