import os
import pandas as pd
import numpy as np
import preprocessing
import clustering
import classes
import metrics as mt
import time

from helper import *

def get_train_data(stock_names, read_path="../sanity_input/train/images_with_labels", labels_are_last=1, save_path="../sanity_input/train/last_saved_data"):
    
    if os.path.isfile(save_path+"/last_saved.pickle"):
        return pd.read_pickle(save_path+"/last_saved.pickle")
    
    all_train_images = []
    all_train_labels = []
    all_train_names = []
    all_train_dates=[]

    # todo: burada şu string colomn işini çöz
    for stock in stock_names:
        data_df = pd.read_csv(read_path + "/{}.csv".format(stock), header=None)
        names = data_df.iloc[:, 0] # first element
        dates = data_df.iloc[:, 1] # second element
        images = data_df.iloc[:, 2:-labels_are_last] # remaining elements
        labels = data_df.iloc[:, -labels_are_last:] # last elements

        print("all images are merging with {} ...".format(stock))

        # determine where to split
        train_image_count = images.shape[0]

        # split train and test
        # for 16 year of data : nearly 14 year train-last 2 year test
        train_images = images.iloc[0:train_image_count]
        train_labels = labels.iloc[0:train_image_count]
        train_names = names.iloc[0:train_image_count]
        train_dates = dates.iloc[0:train_image_count]

        # todo: need to make data class because above not seems good. -ugurgudelek

        if len(all_train_images) == 0:
            all_train_images = np.array(train_images)
            all_train_labels = train_labels.values
            all_train_names = np.array(train_names)
            all_train_dates = np.array(train_dates)
        else:
            all_train_images = np.append(all_train_images, train_images, axis=0)
            all_train_labels = np.append(all_train_labels, train_labels.values, axis=0)
            all_train_names = np.append(all_train_names,train_names, axis=0)
            all_train_dates = np.append(all_train_dates,train_dates, axis=0)

        print("current train shape is {} and {} label ".format(pd.DataFrame(all_train_images).shape,
                                                               all_train_labels.shape[1]))
    
    print("Sorting train data by date and name...")
    sorted_train_data = pd.DataFrame()
    sorted_train_data['date'] = all_train_dates
    sorted_train_data['name'] = all_train_names
    sorted_train_data['image'] = [i for i in all_train_images]
    sorted_train_data['label'] = all_train_labels
    sorted_train_data = sorted_train_data.sort_values(by = ['date', 'name'])
    
    data = {'train_images': pd.DataFrame(np.asarray([i for i in sorted_train_data['image']])),
            'train_labels': pd.DataFrame(sorted_train_data['label'].values),
            'train_names': pd.DataFrame(sorted_train_data['name'].values),
            'train_dates': pd.DataFrame(sorted_train_data['date'].values)
    }    

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pd.to_pickle(data, save_path+"/last_saved.pickle")

    return data

def fast_normalize_and_calculate_metrics(train_stock, fresh_stock):
    train_stock.loc[:,'name'] = [''] * train_stock.shape[0]
    train_stock.loc[:,'pct_change_tanh'] = [0.0] * train_stock.shape[0]

    stock = train_stock.append(fresh_stock, ignore_index=True).drop(['name', 'pct_change_tanh'], axis=1)

    # create data arr to hold all metric info
    metric_data, metric_function_names = preprocessing.calculate_metrics(stock)

    # assign nan value beginning of the data
    metric_data = assign_null_into_data(arr=metric_data, length=len(stock['adjusted_close']))

    # append data and metrics column-wise
    stock = stack_data_and_metrics(stock, metric_data, metric_function_names)

    # normalize price values before applying labels
    # because we need to get rid of diversity among stocks
    stock = preprocessing.apply_normalization_to_raw_data(stock)

    return stock.iloc[-1]


def fast_get_last_image(train_stock_with_metrics, split_period=28):


    predictor_names = pd.read_csv("../sanity_input/train/clustered_names.csv", header=None,
                                  squeeze=True).values.tolist()

    # drop nan values for proper set
    train_stock_with_metrics = train_stock_with_metrics.dropna()

    # drop irrelevant features
    stock_with_metrics = train_stock_with_metrics[['date'] + predictor_names]
    # when i do this, later i can reach data with stock name and date

    image_col_size = stock_with_metrics[predictor_names].shape[1]
    image_row_size = split_period

    if image_row_size != image_col_size:
        raise Exception("image matrix must be square!")

    # lower = num_records - split_period
    # upper = lower + split_period

    image = stock_with_metrics[predictor_names].iloc[-split_period:]
    date = stock_with_metrics['date'].iloc[-split_period:]

    # normalization for image.
    image = (image - image.mean()) / image.std()
    image_flat = image.values.flatten()  # image_flat'shape : image_row_size * image_col_size

    return date, image_flat


def normalize_and_calculate_metrics(stock_name, raw_data_path="../sanity_input/train/raw_data", stock_with_metrics_path = "../sanity_input/train/stock_with_metrics"):
    
    # read stock csv
    stock = pd.read_csv(raw_data_path + "/{}.csv".format(stock_name))

    # read the stock_with_metrics csv
    stock_with_metrics = pd.read_csv(stock_with_metrics_path + "/{}.csv".format(stock_name))

    # get the last 100 records
    num_records = stock.shape[0]
    stock = stock.iloc[num_records - 100 : num_records]
    num_records = stock_with_metrics.shape[0]
    stock_with_metrics = stock_with_metrics.iloc[num_records - 100 : num_records]
    
    # create data arr to hold all metric info
    metric_data, metric_function_names = preprocessing.calculate_metrics(stock)
    
    # assign nan value beginning of the data
    metric_data = assign_null_into_data(arr=metric_data, length=len(stock['adjusted_close']))
    
    # append data and metrics column-wise
    stock = stack_data_and_metrics(stock, metric_data, metric_function_names)
    
    # normalize price values before applying labels
    # because we need to get rid of diversity among stocks
    stock = preprocessing.apply_normalization_to_raw_data(stock)

    # get only the fresh data
    fresh_stock_with_metrics = stock[stock.date.isin(stock_with_metrics.date) == False]
    
    return fresh_stock_with_metrics

def get_last_image(stock_name, split_period=28,
                            stock_with_metrics_path="../sanity_input/train/stock_with_metrics"):

    # read stock_with_metrics
    stock_with_metrics = pd.read_csv(stock_with_metrics_path + "/{}.csv".format(stock_name))

    # get the last 100 records
    # num_records = stock_with_metrics.shape[0]
    # stock_with_metrics = stock_with_metrics.iloc[num_records - 100 : num_records]

    predictor_names = pd.read_csv("../sanity_input/train/clustered_names.csv", header=None, squeeze=True).values.tolist()

    # drop nan values for proper set
    stock_with_metrics = stock_with_metrics.dropna()

    # drop irrelevant features
    stock_with_metrics = stock_with_metrics[['date'] + predictor_names]
    # when i do this, later i can reach data with stock name and date
    

    
    image_col_size = stock_with_metrics[predictor_names].shape[1]
    image_row_size = split_period
    
    if image_row_size != image_col_size:
        raise Exception("image matrix must be square!")

    # lower = num_records - split_period
    # upper = lower + split_period
    
    image = stock_with_metrics[predictor_names].iloc[-split_period:]
    date = stock_with_metrics['date'].iloc[-split_period:]

    # normalization for image.
    image = (image - image.mean()) / image.std()
    image_flat = image.values.flatten()  # image_flat'shape : image_row_size * image_col_size
    
    return date, image_flat
