import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing
import loss_profit
import google_finance_io
import datetime
import os

from keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#import mlp_keras_regr as ms

def main(create_model = False, model_type = "regression"):
    """create_model: a new model is trained and saved if this parameter True
    model_type: type of the model; regression, classification-2, classification-3"""
    
    
    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']

    # todo: fix problem encountered in 'ewu,qqq' etf - ugurgudelek

    if create_model == True:

        # 1.download data
        google_finance_io.download_data(stock_names, start_date=datetime.date(2000, 1, 3),
                                        end_date=datetime.date(2016, 12, 31), verbose=True)

        # 2.calculate metric for available stocks and save them into csv file
        preprocessing.normalize_and_calculate_metrics(stock_names)

        # 4.cluster features for available stocks and their features then save them into csv file
        preprocessing.cluster_features(stock_names, drop_this_cols=['date', 'low', 'close', 'high', 'open','adjusted_close'])

        if model_type == "regression":

            # import our keras regression model
            import cnn_keras_regr as cs

            # 3.calculate labels for available stocks and save them into csv file
            preprocessing.calculate_labels(stock_names)
            
            # 5.read sorted (via hierarchical clustering) feature names from file
            sorted_cluster_names = pd.read_csv("../input/clustered_names.csv", header=None, squeeze=True).values.tolist()
            
            # 6. create flatten images with data and labels.
            preprocessing.create_images_from_data(stock_names, sorted_cluster_names, label_names=['label_day_tanh_regr'])
            
            # 7. merge all available data
            # data has 'images' and 'labels'
            data = preprocessing.get_merged_images_and_labels_data(stock_names, labels_are_last=1, train_test_ratio=0.9)
            
            # 8. call CNN
            params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}
            with K.get_session():
                cs.start_cnn_session(data, params, model_save_name="model_regr_100epoch", model_read_name = "model_regr_100epoch_before_2017_07_19 21_47_40_106461")
            
        # for mlp
        #params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}
        #with K.get_session():
            #ms.start_mlp_session(data, params, model_save_name="model_mlp_regr_100epoch", model_read_name = "") 
        
        elif model_type == "classification-2":

            # import our keras classification-2 model
            import cnn_keras_class as cscls

            preprocessing.calculate_labels(stock_names)
            sorted_cluster_names = pd.read_csv("../input/clustered_names.csv", header=None, squeeze=True).values.tolist()
            preprocessing.create_images_from_data(stock_names, sorted_cluster_names, label_names=['label_day_is_less', 'label_day_is_more'])
            data = preprocessing.get_merged_images_and_labels_data_cls(stock_names, last_image_col = -3, labels_ind = [-2, -1], train_test_ratio = 0.9)
            params = {"input_w": 28, "input_h": 28, "num_classes": 2, "batch_size": 1024, "epochs": 100}
            with K.get_session():
                cscls.start_cnn_session(data, params, model_save_name="model_2class_100epoch", model_read_name = "model_2class_100epoch_before_2017_07_19 22_33_44_785205")

        elif model_type == "classification-3":

            # import our keras classification-3 model
            import cnn_keras_class3 as cscls3

            preprocessing.calculate_labels_3class(stock_names)
            sorted_cluster_names = pd.read_csv("../input/clustered_names.csv", header=None, squeeze=True).values.tolist()
            preprocessing.create_images_from_data(stock_names, sorted_cluster_names, label_names=['label_day_tanh_less', 'label_day_tanh_inrange', 'label_day_tanh_more'])
            data = preprocessing.get_merged_images_and_labels_data_cls(stock_names, last_image_col = -4, labels_ind = [-3, -2, -1], train_test_ratio = 0.9)
            params = {"input_w": 28, "input_h": 28, "num_classes": 3, "batch_size": 1024, "epochs": 100}
            
            with K.get_session():
                cscls3.start_cnn_session(data, params, model_save_name="model_3class_100epoch", model_read_name = "model_3class_100epoch_before_2017_07_19 23_17_01_345010")

    else:

        print("Preparing adjusted close dataframe...")
        prices = loss_profit.prepare_adj_close(stock_names)

        if model_type == "regression":

            print("Calculating final capital using prediction model...")
            capital, shares,_,_ = loss_profit.buy_sell_regr(stock_names, predictions_name = 'predictions_model_regr_100epoch_qratio_0_2017_07_19 22_19_02_006504', adj_close = prices, buy_thr=0, sell_thr=0, transaction_cost=5)

        elif model_type == "classification-2":

            print("Calculating final capital using classification-2 model...")
            capital, shares = loss_profit.buy_sell_class2(predictions_name = 'predictions_model_2class_100epoch_2017_07_19 23_02_34_349364', adj_close = prices, transaction_cost=5)

        elif model_type == "classification-3":

            print("Calculating final capital using classification-3 model...")
            capital, shares = loss_profit.buy_sell_class3(predictions_name = 'predictions_model_3class_100epoch_2017_07_19 23_17_01_345010', adj_close = prices, transaction_cost=5)

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


if __name__ == "__main__":
    
    main(create_model = False, model_type = "classification-3")
    
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

