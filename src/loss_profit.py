
import os
from collections import defaultdict
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
    shares = defaultdict(list)
    # initialize the shares
    for name in stock_names:

        shares[name] = 0

    # find the minimum date among first dates of the predictions
    # predictions are assumed to be sorted by date ascending
    min_date = predictions['Date'].min()
    max_date = predictions['Date'].max()

    # start buying and selling
    current_date = datetime.strptime(min_date, '%Y-%m-%d')
    while current_date < datetime.strptime(max_date, '%Y-%m-%d'):
        
        # find highest stock and it's price on the current date
        higher_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] > 0)]
        
        if not higher_stocks.empty:
            
            highest_stock = predictions.iloc[predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] > 0)]['Prediction'].idxmax()][['Name', 'Prediction']]
            highest_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == highest_stock['Name'])]['Adj_Close'])
        
            # if we have enough capital to buy the highest_stock, buy it with all of our money
            if capital > highest_price:
    
                print("Stock {} will go up ({}), buying...".format(highest_stock['Name'], highest_stock['Prediction']))
                shares[highest_stock['Name']] += capital // highest_price
                capital -= (capital // highest_price) * highest_price
        
        lower_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] < 0)][['Name', 'Prediction']]
                
        # if lower_stocks is not empty
        if not lower_stocks.empty:

            # for each stock in lower_stocks
            for lower_stock in lower_stocks['Name']:

                # if we have any of this stock
                if shares[lower_stock] != 0:
                    
                    # find the price of this stock on the current date
                    lower_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == lower_stock)]['Adj_Close'])

                    # sell this stock
                    print("Stocks {} will go down ({}), selling...".format(lower_stock, lower_stocks.loc[lower_stocks['Name'] == lower_stock]['Prediction']))
                    capital += lower_price * shares[lower_stock]
                    shares[lower_stock] = 0

        print("Date: " + current_date.strftime("%Y-%m-%d"))
        print("Capital: " + str(capital))
        print("Shares: " + str(shares))
        print("----------------------------")

        # increment date by 1 day
        current_date += timedelta(days = 1)

        # sell all shares to obtain the final capital at the end of the term
        if current_date == datetime.strptime(max_date, '%Y-%m-%d') and shares:

            print("End of the term, selling all the shares...")
            
            for current_share in shares:
                
                # price of the share on the current date
                current_share_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == current_share)]['Adj_Close'])
                
                # sell this stock
                capital += current_share_price * shares[current_share]
                shares[current_share] = 0

            print("Date: " + current_date.strftime("%Y-%m-%d"))
            print("Capital: " + str(capital))
            print("Shares: " + str(shares))
            print("----------------------------")
                            
    return capital, shares