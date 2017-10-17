
import numpy as np
import csv
import pandas as pd
import sanity_testmodule
import sanity_preprocessing
import datetime
import os
import shutil
import glob

from keras.models import load_model


"""THIS MODULE UPDATES THE stock_with_metrics OF sanity INPUTS.
FOR EACH GIVEN DAILY DATA FOR A STOCK, IT CALCULATES THE NEW METRICS AND CALCULATES THE NEW IMAGE.
THEN, IT LOADS THE MODEL AND MAKES PREDICTIONS ABOUT STOCK PRICES USING THE NEW IMAGES AND RETURNS
THE PREDICTIONS.
"""
stock_names = ['spy', 'xlf', 'xlu', 'xle',
               'xlp', 'xli', 'xlv', 'xlk', 'ewj',
               'xlb', 'xly', 'eww', 'dia', 'ewg',
               'ewh', 'ewc', 'ewa']
raw_data_path = "../sanity_input/train/raw_data"
stock_with_metrics_path = "../sanity_input/train/stock_with_metrics"
model_path = "../sanity_model"

# model parameters
params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}

model = load_model(model_path + "/model_regr_100epoch_before_2017_10_17 11_54_10_765826")
# read the stock info from local database
train_stocks = dict()
train_stocks_with_metrics = dict()

for stock_name in stock_names:
    train_stocks[stock_name] = pd.read_csv(raw_data_path + "/{}.csv".format(stock_name))
    train_stocks_with_metrics[stock_name] = pd.read_csv(stock_with_metrics_path + "/{}.csv".format(stock_name))



def update_data_and_predict(next_day_data):

    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']

    raw_data_path = "../sanity_input/train/raw_data"
    stock_with_metrics_path = "../sanity_input/train/stock_with_metrics"
    model_path = "../sanity_model"

    predictions_dict = {}

    # load the latest file as model
    # list_of_files = glob.glob("{}/*".format(model_path))
    # latest_file = max(list_of_files, key=os.path.getctime)
    # model = load_model(latest_file)

    # model = load_model(model_path + "/model_regr_100epoch_before_2017_10_16 16_37_31_758941")

    # model parameters
    params = {"input_w": 28, "input_h": 28, "num_classes": 1, "batch_size": 1024, "epochs": 100}
    
    # update data for each stock (except for last_saved_data)
    for stock_name in stock_names:
        
        # read the stock info from local database
        stock = pd.read_csv(raw_data_path + "/{}.csv".format(stock_name))
        
        # get the lastest stock info
        fresh_stock = next_day_data[stock_name]

        filename = raw_data_path + "/{}.csv".format(stock_name)
        old_stock = pd.read_csv(filename)
        old_stock['name'] = ['']* old_stock.shape[0]
        old_stock['pct_change_tanh'] = [0.0] * old_stock.shape[0]

        old_stock = old_stock.append(fresh_stock, ignore_index=True)
        old_stock = old_stock.drop(['name','pct_change_tanh'],axis=1)

        # append it to raw_data
        # with open(raw_data_path + "/{}.csv".format(stock_name), 'a') as f:
        old_stock.to_csv(raw_data_path + "/{}.csv".format(stock_name), index = None)
            
        # calculate the metrics for the new data
        fresh_stock_with_metrics = sanity_preprocessing.normalize_and_calculate_metrics(stock_name, raw_data_path, stock_with_metrics_path)
        # append the new metrics data to stock_with_metrics
        filename = stock_with_metrics_path + "/{}.csv".format(stock_name)
        old_stock = pd.read_csv(filename)
        old_stock.append(fresh_stock_with_metrics).to_csv(stock_with_metrics_path + "/{}.csv".format(stock_name), index = None)
        
        # calculate images for the new data
        [fresh_stock_image_date, fresh_stock_image] = sanity_preprocessing.get_last_image(stock_name, stock_with_metrics_path = stock_with_metrics_path)

        # make the image square
        image = fresh_stock_image.reshape((1, params["input_w"], params["input_h"], 1))


        # make the prediction and record to the dictionary
        prediction = model.predict(image)
        predictions_dict[stock_name] = prediction[0][0]


    return predictions_dict

def fast_update_data_and_predict(next_day_data):
    predictions_dict = {}
    images_dict = {}

    # load the latest file as model
    # list_of_files = glob.glob("{}/*".format(model_path))
    # latest_file = max(list_of_files, key=os.path.getctime)
    # model = load_model(latest_file)

    # update data for each stock (except for last_saved_data)
    for stock_name in stock_names:

        # if loaded_flag[stock_name] == 0:
        train_stock = train_stocks[stock_name]
        train_stock_with_metrics = train_stocks_with_metrics[stock_name]

        # get the lastest stock info
        fresh_stock = next_day_data[stock_name]


        # calculate the metrics for the new data
        fresh_stock_with_metrics = sanity_preprocessing.fast_normalize_and_calculate_metrics(train_stock, fresh_stock)

        # update data
        train_stock = train_stock.append(fresh_stock).reset_index(drop=True)
        train_stock_with_metrics = train_stock_with_metrics.append(fresh_stock_with_metrics).reset_index(drop=True)

        # calculate images for the new data
        [fresh_stock_image_date, fresh_stock_image] = sanity_preprocessing.fast_get_last_image(train_stock_with_metrics)

        # print('Name:{} Date:{}'.format(stock_name, train_stock.iloc[-1].date))
        # make the image square
        image = fresh_stock_image.reshape((1, params["input_w"], params["input_h"], 1))
        images_dict[stock_name] = image

        # make the prediction and record to the dictionary
        prediction = model.predict(image)
        predictions_dict[stock_name] = prediction[0][0]

    return (predictions_dict,images_dict)

def update_model(images, labels):
    for stock_name in stock_names:
        image = images[stock_name]
        label = labels[stock_name]
        image = image.reshape((1, params["input_w"], params["input_h"], 1))
        label = label.reshape((1, params["num_classes"]))
        # train with only 1 more image
        model.train_on_batch(image, label)
    
