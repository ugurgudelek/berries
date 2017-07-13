
import os
import numpy as np
import pandas as pd
import copy
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
        

def buy_sell_regr(predictions_name, adj_close, initial_capital = 10000, predictions_path = "../result/"):
    """This function buys and sells stocks for regression according to given thresholds.
    predictions_name: name of the file that contains the predictions data.
    adj_close: adjusted closes for stocks. This is a dataframe indexed by stock names, like predictions."""
    
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
    min_date = predictions['Date'].min()
    max_date = predictions['Date'].max()

    # the elements of each operation is recorded to below lists, which will be converted to DataFrame finally
    record_dates = []
    record_operations = []
    record_names = []
    record_amounts = []
    record_prices = []
    record_capitals = []

    # amount of each share after an operation, thus, have the same amount of rows as the lists above
    record_shares = []

    # start buying and selling
    current_date = datetime.strptime(min_date, '%Y-%m-%d')
    while current_date < datetime.strptime(max_date, '%Y-%m-%d'):
        
        # find highest stock and it's price on the current date
        higher_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] > buy_thr)]
        
        if not higher_stocks.empty:
            
            highest_stock = predictions.iloc[predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] > 0)]['Prediction'].idxmax()][['Name', 'Prediction']]
            highest_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == highest_stock['Name'])]['Adj_Close'])
        
            # if we have enough capital to buy the highest_stock, buy it with all of our money
            if capital > highest_price and type(highest_price) is float:
    
                print("Stock {} will go up ({}), buying...".format(highest_stock['Name'], highest_stock['Prediction']))
                highest_amount = capital // highest_price
                shares[highest_stock['Name']] += highest_amount
                capital -= highest_amount * highest_price

                record_dates.append(current_date.strftime("%Y-%m-%d"))
                record_operations.append('buy')
                record_names.append(highest_stock['Name'])
                record_amounts.append(highest_amount)
                record_prices.append(highest_price)
                record_capitals.append(capital)
                record_shares.append(copy.copy(shares))
        
        lower_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] < sell_thr)][['Name', 'Prediction']]
                
        # if lower_stocks is not empty
        if not lower_stocks.empty:

            # for each stock in lower_stocks
            for lower_stock in lower_stocks['Name']:

                # if we have any of this stock
                if shares[lower_stock] > 0:
                    
                    # find the price of this stock on the current date
                    lower_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == lower_stock)]['Adj_Close'])

                    # sell this stock
                    print("Stocks {} will go down ({}), selling...".format(lower_stock, lower_stocks.loc[lower_stocks['Name'] == lower_stock]['Prediction']))
                    lower_amount = shares[lower_stock]
                    capital += lower_price * lower_amount
                    shares[lower_stock] = 0

                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('sell')
                    record_names.append(lower_stock)
                    record_amounts.append(lower_amount)
                    record_prices.append(lower_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))

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
                current_amount = shares[current_share]
                capital += current_share_price * current_amount
                shares[current_share] = 0

                record_dates.append(current_date.strftime("%Y-%m-%d"))
                record_operations.append('sell')
                record_names.append(current_share)
                record_amounts.append(current_amount)
                record_prices.append(current_share_price)
                record_capitals.append(capital)
                record_shares.append(copy.copy(shares))

            print("Date: " + current_date.strftime("%Y-%m-%d"))
            print("Capital: " + str(capital))
            print("Shares: " + str(shares))
            print("----------------------------")

    # save records to file
    now = str(datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')
    pd.DataFrame({'Dates':record_dates, 'Operations':record_operations, 'Names':record_names, 'Amounts':record_amounts, 'Prices':record_prices, 'Capitals':record_capitals}).to_pickle(predictions_path + "buy_sell_model_regr_" + now)
    pd.DataFrame.from_dict(record_shares).to_pickle(predictions_path + "buy_sell_model_regr_shares_" + now)
                            
    return capital, shares

def buy_hold(stock_names, adj_close, initial_capital = 10000, start_date = "2015-03-24"):
    """This function buys stocks spending equal amount of money for each one (if possible)
    and sells all of the stocks at the end of the term."""

    # again, we're assuming that adjusted close values are sorted in ascending order by date
    
    num_of_stocks = len(stock_names)
    money_per_stock = initial_capital / num_of_stocks
    capital = initial_capital

    print("Money for each stock:")
    print(money_per_stock)

    # buy shares for each stock with equal amount of money
    shares = {}    
    for stock in stock_names:

        stock_start_price = float(adj_close.loc[np.logical_and(adj_close['Name'] == stock, adj_close['Date'] == start_date)]['Adj_Close'])
        stock_amount = money_per_stock // stock_start_price
        
        if stock_amount > 0:

            print("Buying {} amount of stock {} at price {}".format(str(stock_amount), stock, str(stock_start_price)))
            
            # buy as much shares as stock_amount for the current stock
            shares[stock] = stock_amount
            # update the capital
            capital -= stock_amount * stock_start_price

    print("Capital left:")
    print(capital)

    print("----------------------------------------------------")

    # sell the shares we have at the end of the term
    for stock in stock_names:

        # if we have any shares for that stock
        if shares[stock] > 0:

            # find the last date for the stock in adjusted close and the price on that date
            stock_last_date = adj_close.loc[adj_close['Name'] == stock]['Date'].iloc[-1]
            stock_last_price = float(adj_close.loc[np.logical_and(adj_close['Name'] == stock, adj_close['Date'] == stock_last_date)]['Adj_Close'])

            print("Selling {} amount of stock {} at price {}".format(str(shares[stock]), stock, str(stock_last_price)))

            # sell the shares
            capital += shares[stock] * stock_last_price
            shares[stock] = 0

    return capital, shares
    
