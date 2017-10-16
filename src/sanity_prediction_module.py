
import numpy as np
import csv
import pandas as pd
import sanity_testmodule
import sanity_preprocessing
import datetime
import os
import shutil
import glob


"""THIS MODULE UPDATES THE stock_with_metrics OF sanity INPUTS.
FOR EACH GIVEN DAILY DATA FOR A STOCK, IT CALCULATES THE NEW METRICS AND CALCULATES THE NEW IMAGE.
THEN, IT LOADS THE MODEL AND MAKES PREDICTIONS ABOUT STOCK PRICES USING THE NEW IMAGES AND RETURNS
THE PREDICTIONS.
"""

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
        # todo: fix here
        pd.concat((old_stock, fresh_stock), axis=1)

        # append it to raw_data
        with open(raw_data_path + "/{}.csv".format(stock_name), 'a') as f:
            fresh_stock.to_csv(f, header = False, index = None)
            
        # calculate the metrics for the new data
        fresh_stock_with_metrics = sanity_preprocessing.normalize_and_calculate_metrics(stock_name, raw_data_path)
        # append the new metrics data to stock_with_metrics
        with open(stock_with_metrics_path + "/{}.csv".format(stock_name), 'a') as f:
            fresh_stock_with_metrics.to_csv(f, header = False, index = None)
        
        # calculate images for the new data
        [fresh_stock_image_date, fresh_stock_image] = sanity_preprocessing.get_last_image(stock_name, label_names = ["label_day_tanh_regr"], stock_with_metrics_path = stock_with_metrics_path)

        # make the image square
        image = fresh_stock_image.reshape((1, params["input_w"], params["input_h"], 1))

        # make the prediction and record to the dictionary
        prediction = model.predict(image)
        predictions_dict[stock_name] = prediction[0][0]

    return predictions_dict
    
