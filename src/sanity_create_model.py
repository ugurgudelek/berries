import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import preprocessing
import sanity_cnn_train as sncs
import sanity_preprocessing
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K



def create_model():
    
    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']
       
    # 2.calculate metric for available stocks and save them into csv file
    preprocessing.normalize_and_calculate_metrics(stock_names, raw_data_path="../sanity_input/train/raw_data", path_to_save="../sanity_input/train/stock_with_metrics")
    
    # 4.cluster features for available stocks and their features then save them into csv file
    preprocessing.cluster_features(stock_names, drop_this_cols=['date', 'low', 'close', 'high', 'open','adjusted_close'], \
        stock_with_metric_path="../sanity_input/train/stock_with_metrics", save_path="../sanity_input/train")
    
    # 3.calculate labels for available stocks and save them into csv file
    preprocessing.calculate_labels(stock_names, stock_with_metric_path="../sanity_input/train/stock_with_metrics",
                                   stock_with_labels_path="../sanity_input/train/stock_with_labels")
    
    # 5.read sorted (via hierarchical clustering) feature names from file
    sorted_cluster_names = pd.read_csv("../sanity_input/train/clustered_names.csv", header=None, squeeze=True).values.tolist()
    
    # 6. create flatten images with data and labels.
    preprocessing.create_images_from_data(stock_names, sorted_cluster_names, label_names=['label_day_tanh_regr'], stock_with_labels_path="../sanity_input/train/stock_with_labels",
                            save_path="../sanity_input/train/images_with_labels")
    
    # 7. merge all available data
    # data has 'images' and 'labels'
    data = sanity_preprocessing.get_train_data(stock_names, labels_are_last=1,  read_path="../sanity_input/train/images_with_labels", save_path="../sanity_input/train/last_saved_data")
    
    # 8. call CNN
    params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}
    with K.get_session():
        sncs.start_cnn_session(data, params, model_save_name="model_regr_100epoch", model_read_name = "")

if __name__ == "__main__":
    create_model()
