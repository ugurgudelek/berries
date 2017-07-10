
import os
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


def prepare_adj_close(stock_names, raw_data_path = "../input/raw_data"):

    """This function reads raw data and prepares adjusted close dataframe, which is indexed by stock_names.
    The first element of each row is the date and the second element is adjusted close value on that date."""

    adj_close = {}

    # populate the dictionary
    for name in stock_names:

        if name + ".csv" not in os.listdir(raw_data_path):
            
            print("Cannot find {}.csv in the specified path..".format(name))
            return None

        stock = pd.read_csv(raw_data_path + "/{}.csv".format(name))

        stock_data = np.concatenate((stock['date'].values.reshape((-1,1)), stock['adjusted_close'].values.reshape((-1,1))), axis = 1)

        adj_close[name] = stock_data

    adj_close = pd.Panel(adj_close).to_frame()

    return adj_close
        

def buy_sell_regr(predictions_name, adj_close, initial_captial = 10000, buy_thr = .38, sell_thr = -.38, predictions_path = "../result/"):
    """This function buys and sells stocks for regression according to given thresholds.
    predictions_name: name of the file that contains the predictions data.
    adj_close: adjusted closes for stocks. This is a dataframe indexed by stock names, like predictions.
    buy_thr: if the sigmoid values are predicted to exceed this threshold, then buy the highest predicted stock.
    sell_thr: if the sigmoid values are predicted to fall behind this threshold, then sell the lowest predicted stock."""

    # read the predictions data
    predictions = pd.read_pickle(predictions_path + predictions_name)
    # stock names
    stock_names = predictions.keys()
    # our captial
    captial = initial_captial
    # number of shares for each stock
    shares = {}
    # initialize the shares
    for name in stock_names:

        shares[name] = 0

    # find the minimum date among first dates of the predictions
    # predictions are assumed to be sorted by date ascending
    date_fmt = '%Y-%m-%d'
    min_date = datetime.strptime('3000-12-12', date_fmt)
    max_date = datetime.strptime('1000-12-12', date_fmt)
    for name in stock_names:

        tmp_date = datetime.strptime(predictions[name][0][0], date_fmt)
        
        if tmp_date < min_date:

            min_date = tmp_date

        if tmp_date > max_date:

            max_date = tmp_date


    # start buying and selling
    current_date = min_date
    while current_date < max_date :

        # stocks that we have predicted to go higher from this date and how high we predict
        higher_stocks = {}
        # stocks that we have predicted to go lower from this date
        lower_stocks = []

        # find higher and lower stocks
        for name in stock_names:

            for i in range(0, predictions[name].shape[0] / predictions[name][0].shape[0]):

                tmp_date = datetime.strptime(predictions[name][i][0], date_fmt)

                if tmp_date == current_date:

                    if float(predictions[name][i][1]) > 0 :

                        higher_stocks[name] = float(predictions[name][i][1])

                    elif float(predictions[name][i][1]) < 0 :

                        lower_stocks.append(name)
                        
                    break  # we can break because predictions are assumed to be sorted by date ascending

        # buy - sell operation

        # if higher_stocks is not empty
        if not higher_stocks:

            highest_stock = None
            highest_up_change = 0

            # find the highest stock
            for cur_high_stock in higher_stocks.keys():

                if higher_stocks[cur_high_stock] > highest_up_change:

                    highest_stock = cur_high_stock
                    highest_up_change = higher_stocks[cur_high_stock]
            

            # find the price of the highest stock on the current date
            highest_price = 0
            for i in range(0, adj_close[highest_stock].shape[0] / adj_close[highest_stock][0].shape[0]):
                
                if datetime.strptime(adj_close[highest_stock][i][0], date_fmt) == current_date:
                    
                    highest_price = float(adj_close[highest_stock][i][1])
                    break
                
            # if we have enough capital to buy the highest_stock, buy it with all of our money
            if captial > highest_price:

                shares[highest_stock] += captial // highest_price
                
        # if lower_stocks is not empty
        if not lower_stocks:

            # for each stock in lower_stocks
            for lower_stock in lower_stocks:

                # if we have any of this stock
                if shares[lower_stock] != 0:
                    
                    # find the price of this stock on the current date
                    lower_price = 0
                    for i in range(0, adj_close[lower_stock].shape[0] / adj_close[lower_stock][0].shape[0]):
                
                        if datetime.strptime(adj_close[lower_stock][i][0], date_fmt) == current_date:
                    
                            lower_price = float(adj_close[lower_stock][i][1])
                            break

                    # sell this stock
                    captial += lower_price * shares[lower_stock]
                    shares[lower_stock] = 0

        # increment date by 1 day
        current_date += timedelta(days = 1)

    return captial, shares
            
        
