
import os
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta


def prepare_adj_close(stock_names, raw_data_path = "../input/raw_data"):

    """This function reads raw data and prepares adjusted close dataframe, which is indexed by stock_names.
    The first element of each row is the date and the second element is adjusted close value on that date."""

    closes = None
    names = None
    dates = None

    # populate the dictionary
    for name in stock_names:

        if name + ".csv" not in os.listdir(raw_data_path):
            
            print("Cannot find {}.csv in the specified path..".format(name))
            return None

        stock = pd.read_csv(raw_data_path + "/{}.csv".format(name))
        
        if closes is None:
            closes = stock['adjusted_close'].values
            dates = stock['date'].values
            names = np.repeat(name, stock['date'].values.shape[0])
        else:
            closes = np.concatenate((closes, stock['adjusted_close'].values))
            dates = np.concatenate((dates, stock['date'].values))
            names = np.concatenate((names, np.repeat(name, stock['date'].values.shape[0])))

    adj_close = pd.DataFrame({'Name' : names, 'Date' : dates, 'Adj_Close' : closes})

    return adj_close
        

def buy_sell_regr(predictions_name, adj_close, initial_capital = 10000, buy_thr = .38, sell_thr = -.38, predictions_path = "../result/"):
    """This function buys and sells stocks for regression according to given thresholds.
    predictions_name: name of the file that contains the predictions data.
    adj_close: adjusted closes for stocks. This is a dataframe indexed by stock names, like predictions.
    buy_thr: if the sigmoid values are predicted to exceed this threshold, then buy the highest predicted stock.
    sell_thr: if the sigmoid values are predicted to fall behind this threshold, then sell the lowest predicted stock."""
    
    # read the predictions data
    predictions = pd.read_pickle(predictions_path + predictions_name)
    # stock names
    stock_names = predictions['Name'].unique()
    # our capital
    capital = initial_capital
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

        pred_length = int(predictions.loc[predictions['Name'] == name].shape[0])
        
        tmp_min_date = datetime.strptime(predictions.loc[predictions['Name'] == name].iloc[0]['Date'], date_fmt)
        tmp_max_date = datetime.strptime(predictions.loc[predictions['Name'] == name].iloc[pred_length - 1]['Date'], date_fmt)
        
        if tmp_min_date < min_date:

            min_date = tmp_min_date

        if tmp_max_date > max_date:

            max_date = tmp_max_date


    # start buying and selling
    current_date = min_date
    while current_date < max_date :

        # stocks that we have predicted to go higher from this date and how high we predict
        higher_stocks = {}
        # stocks that we have predicted to go lower from this date
        lower_stocks = {}

        # find higher and lower stocks
        for name in stock_names:

            for i in range(0, int(predictions.loc[predictions['Name'] == name].shape[0])):

                tmp_date = datetime.strptime(predictions.loc[predictions['Name'] == name].iloc[i]['Date'], date_fmt)

                if tmp_date == current_date:
                
                    tmp_prediction = float(predictions.loc[predictions['Name'] == name].iloc[i]['Prediction'])

                    if tmp_prediction > 0 :

                        higher_stocks[name] = tmp_prediction

                    elif tmp_prediction < 0 :

                        lower_stocks[name] = tmp_prediction
                        
                    break  # we can break because predictions are assumed to be sorted by date ascending

        # buy - sell operation

        # if higher_stocks is not empty
        if higher_stocks:

            highest_stock = None
            highest_up_change = 0

            # find the highest stock
            for cur_high_stock in higher_stocks:

                if higher_stocks[cur_high_stock] > highest_up_change:

                    highest_stock = cur_high_stock
                    highest_up_change = higher_stocks[cur_high_stock]
            

            # find the price of the highest stock on the current date
            highest_price = 0
            for i in range(0, int(adj_close.loc[adj_close['Name'] == highest_stock].shape[0])):
                
                if datetime.strptime(adj_close.loc[adj_close['Name'] == highest_stock].iloc[i]['Date'], date_fmt) == current_date:
                    
                    highest_price = float(adj_close.loc[adj_close['Name'] == highest_stock].iloc[i]['Adj_Close'])
                    break
                
            # if we have enough capital to buy the highest_stock, buy it with all of our money
            if capital > highest_price:

                print("Stock {} will go up ({}), buying...".format(highest_stock, higher_stocks[highest_stock]))
                shares[highest_stock] += capital // highest_price
                capital -= (capital // highest_price) * highest_price
                
        # if lower_stocks is not empty
        if lower_stocks:

            # for each stock in lower_stocks
            for lower_stock in lower_stocks:

                # if we have any of this stock
                if shares[lower_stock] != 0:
                    
                    # find the price of this stock on the current date
                    lower_price = 0
                    for i in range(0, int(adj_close.loc[adj_close['Name'] == lower_stock].shape[0])):
                
                        if datetime.strptime(adj_close.loc[adj_close['Name'] == lower_stock].iloc[i]['Date'], date_fmt) == current_date:
                    
                            lower_price = float(adj_close.loc[adj_close['Name'] == lower_stock].iloc[i]['Adj_Close'])
                            break

                    # sell this stock
                    print("Stocks {} will go down ({}), selling...".format(lower_stock, lower_stocks[lower_stock]))
                    capital += lower_price * shares[lower_stock]
                    shares[lower_stock] = 0

        print("Date: " + current_date.strftime(date_fmt))
        print("Capital: " + str(capital))
        print("Shares: " + str(shares))
        print("----------------------------")

        # increment date by 1 day
        current_date += timedelta(days = 1)

        # sell higher shares to obtain the final capital at the end of the term
        if current_date == max_date:

            print("End of the term, selling all higher stocks..")
            
            # if higher_stocks is not empty
            if higher_stocks:
                
                # for each stock in higher_stocks
                for higher_stock in higher_stocks:
                    
                    # if we have any of this stock
                    if shares[higher_stock] != 0:
                        
                        # find the price of this stock on the current date
                        higher_price = 0
                        for i in range(0, int(adj_close.loc[adj_close['Name'] == higher_stock].shape[0])):
                            
                            if datetime.strptime(adj_close.loc[adj_close['Name'] == higher_stock].iloc[i]['Date'], date_fmt) == current_date:
                                
                                higher_price = float(adj_close.loc[adj_close['Name'] == higher_stock].iloc[i]['Adj_Close'])
                                break
                            
                        # sell this stock
                        capital += higher_price * shares[higher_stock]
                        shares[higher_stock] = 0


            print("Date: " + current_date.strftime(date_fmt))
            print("Capital: " + str(capital))
            print("Shares: " + str(shares))
            print("----------------------------")
                            
    return capital, shares