
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
        

def buy_sell_regr(stock_names, predictions_name, adj_close, initial_capital = 10000.0, buy_thr= 0.0, sell_thr=0.0,transaction_cost = 5, predictions_path = "../result/", verbose=True):
    """This function buys and sells stocks for regression according to given thresholds.
    predictions_name: name of the file that contains the predictions data.
    adj_close: adjusted closes for stocks. This is a dataframe indexed by stock names, like predictions."""



    # read the predictions data
    predictions = pd.read_pickle(predictions_path + predictions_name)

    # select only those stated in stock_names parameter
    idxs = [list(predictions.loc[predictions['Name'] == stock_name].index) for stock_name in stock_names]  # this is list of lists
    idxs_fixed = [i for idx in idxs for i in idx] # now it is a list
    # update predictions
    predictions = predictions.iloc[idxs_fixed].reset_index(drop = True)

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
    record_transation_costs = []

    # amount of each share after an operation, thus, have the same amount of rows as the lists above
    record_shares = []

    # start buying and selling
    current_date = datetime.strptime(min_date, '%Y-%m-%d')
    while current_date < datetime.strptime(max_date, '%Y-%m-%d'):
        
        # find highest stock and it's price on the current date
        higher_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] > buy_thr)]
        
        if not higher_stocks.empty:
            
            highest_stock = predictions.iloc[predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] > 0)]['Prediction'].idxmax()][['Name', 'Prediction']]
            highest_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == highest_stock['Name'])]['Adj_Close'].iloc[0])
        
            # if we have enough capital to buy the highest_stock, buy it with all of our money
            if capital > highest_price + transaction_cost and type(highest_price) is float:
                if verbose:
                    print("Stock {} will go up ({}), buying...".format(highest_stock['Name'], highest_stock['Prediction']))
                highest_amount = (capital - transaction_cost) // highest_price
                shares[highest_stock['Name']] += highest_amount
                capital -= highest_amount * highest_price
                capital -= transaction_cost  # sub transaction cost


                record_dates.append(current_date.strftime("%Y-%m-%d"))
                record_operations.append('buy')
                record_names.append(highest_stock['Name'])
                record_amounts.append(highest_amount)
                record_prices.append(highest_price)
                record_capitals.append(capital)
                record_shares.append(copy.copy(shares))
                record_transation_costs.append(transaction_cost)
        
        lower_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), predictions['Prediction'] < sell_thr)][['Name', 'Prediction']]
                
        # if lower_stocks is not empty
        if not lower_stocks.empty:

            # for each stock in lower_stocks
            for lower_stock in lower_stocks['Name']:

                # if we have any of this stock
                if shares[lower_stock] > 0:
                    
                    # find the price of this stock on the current date
                    lower_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == lower_stock)]['Adj_Close'].iloc[0])


                    # sell this stock
                    if verbose:
                        print("Stocks {} will go down ({}), selling...".format(lower_stock, lower_stocks.loc[lower_stocks['Name'] == lower_stock]['Prediction']))
                    lower_amount = shares[lower_stock]
                    capital += lower_price * lower_amount
                    shares[lower_stock] = 0
                    capital -= transaction_cost #sub transaction cost

                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('sell')
                    record_names.append(lower_stock)
                    record_amounts.append(lower_amount)
                    record_prices.append(lower_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)

        if verbose:
            print("Date: " + current_date.strftime("%Y-%m-%d"))
            print("Capital: " + str(capital))
            print("Shares: " + str(shares))
            print("----------------------------")

        # increment date by 1 day
        current_date += timedelta(days = 1)

        # sell all shares to obtain the final capital at the end of the term
        if current_date == datetime.strptime(max_date, '%Y-%m-%d') and shares:
            if verbose:
                print("End of the term, selling all the shares...")
            
            for current_share in shares:
                
                # price of the share on the current date
                current_share_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == current_share)]['Adj_Close'].iloc[0])
                
                # sell this stock
                current_amount = shares[current_share]
                if current_amount > 0: #if we have this share
                    capital += current_share_price * current_amount
                    shares[current_share] = 0
                    capital -= transaction_cost

                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('sell')
                    record_names.append(current_share)
                    record_amounts.append(current_amount)
                    record_prices.append(current_share_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)
            if verbose:
                print("Date: " + current_date.strftime("%Y-%m-%d"))
                print("Capital: " + str(capital))
                print("Shares: " + str(shares))
                print("----------------------------")

    # save records to file
    now = str(datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')

    record_transactions_df = pd.DataFrame({'Dates':record_dates, 'Operations':record_operations, 'Names':record_names, 'Amounts':record_amounts, 'Prices':record_prices, 'Capitals':record_capitals, 'Transaction Costs':record_transation_costs})
    record_shares_df = pd.DataFrame.from_dict(record_shares)

    record_transactions_df.to_pickle(predictions_path + "buy_sell_model_regr_" + now)
    record_shares_df.to_pickle(predictions_path + "buy_sell_model_regr_shares_" + now)
                            
    return capital, shares, record_transactions_df, record_shares_df

def buy_hold(stock_names, adj_close, initial_capital = 10000, start_date = "", verbose=True):
    """This function buys stocks spending equal amount of money for each one (if possible)
    and sells all of the stocks at the end of the term."""

    # again, we're assuming that adjusted close values are sorted in ascending order by date
    
    num_of_stocks = len(stock_names)
    money_per_stock = initial_capital / num_of_stocks
    capital = initial_capital

    if verbose:
        print("Money for each stock:")
        print(money_per_stock)

    # buy shares for each stock with equal amount of money
    shares = {}    
    for stock in stock_names:

        stock_start_price = float(adj_close.loc[np.logical_and(adj_close['Name'] == stock, adj_close['Date'] == start_date)]['Adj_Close'].iloc[0])
        stock_amount = money_per_stock // stock_start_price
        
        if stock_amount > 0:
            if verbose:
                print("Buying {} amount of stock {} at price {}".format(str(stock_amount), stock, str(stock_start_price)))
            
            # buy as much shares as stock_amount for the current stock
            shares[stock] = stock_amount
            # update the capital
            capital -= stock_amount * stock_start_price
    if verbose:
        print("Capital left:")
        print(capital)

        print("----------------------------------------------------")

    # sell the shares we have at the end of the term
    for stock in stock_names:

        # if we have any shares for that stock
        if shares[stock] > 0:

            # find the last date for the stock in adjusted close and the price on that date
            stock_last_date = adj_close.loc[adj_close['Name'] == stock]['Date'].iloc[-1]
            stock_last_price = float(adj_close.loc[np.logical_and(adj_close['Name'] == stock, adj_close['Date'] == stock_last_date)]['Adj_Close'].iloc[0])
            if verbose:
                print("Selling {} amount of stock {} at price {}".format(str(shares[stock]), stock, str(stock_last_price)))

            # sell the shares
            capital += shares[stock] * stock_last_price
            shares[stock] = 0

    return capital, shares

def buy_sell_class2(predictions_name, adj_close, initial_capital = 10000.0, transaction_cost = 5, predictions_path = "../result/"):
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
    record_transation_costs = []

    # amount of each share after an operation, thus, have the same amount of rows as the lists above
    record_shares = []

    # start buying and selling
    current_date = datetime.strptime(min_date, '%Y-%m-%d')
    while current_date < datetime.strptime(max_date, '%Y-%m-%d'):
        
        # find higher stocks and their prices on the current date
        higher_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), np.argmax(np.concatenate((predictions['Pred0'].values.reshape(-1,1), predictions['Pred1'].values.reshape(-1,1)), axis = 1), axis = 1) == 1)]
        
        if not higher_stocks.empty:

            for higher_stock in higher_stocks['Name']:
            
                higher_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == higher_stock)]['Adj_Close'].iloc[0])
            
                # if we have enough capital to buy the higher_stock, buy it with all of our money
                if capital > higher_price + transaction_cost:
                
                    print("Stock {} will go up, buying...".format(higher_stock))
                    higher_amount = (capital - transaction_cost) // higher_price
                    shares[higher_stock] += higher_amount
                    capital -= higher_amount * higher_price
                    capital -= transaction_cost  # sub transaction cost                    
                    
                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('buy')
                    record_names.append(higher_stock)
                    record_amounts.append(higher_amount)
                    record_prices.append(higher_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)
        
        lower_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), np.argmax(np.concatenate((predictions['Pred0'].values.reshape(-1,1), predictions['Pred1'].values.reshape(-1,1)), axis = 1), axis = 1) == 0)]['Name']
                
        # if lower_stocks is not empty
        if not lower_stocks.empty:

            # for each stock in lower_stocks
            for lower_stock in lower_stocks:

                # if we have any of this stock
                if shares[lower_stock] > 0:
                    
                    # find the price of this stock on the current date
                    lower_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == lower_stock)]['Adj_Close'].iloc[0])

                    # sell this stock
                    print("Stocks {} will go down, selling...".format(lower_stock))
                    lower_amount = shares[lower_stock]
                    capital += lower_price * lower_amount
                    shares[lower_stock] = 0
                    capital -= transaction_cost #sub transaction cost

                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('sell')
                    record_names.append(lower_stock)
                    record_amounts.append(lower_amount)
                    record_prices.append(lower_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)

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
                current_share_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == current_share)]['Adj_Close'].iloc[0])
                
                # sell this stock
                current_amount = shares[current_share]
                if current_amount > 0: #if we have this share
                    capital += current_share_price * current_amount
                    shares[current_share] = 0
                    capital -= transaction_cost

                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('sell')
                    record_names.append(current_share)
                    record_amounts.append(current_amount)
                    record_prices.append(current_share_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)

            print("Date: " + current_date.strftime("%Y-%m-%d"))
            print("Capital: " + str(capital))
            print("Shares: " + str(shares))
            print("----------------------------")

    # save records to file
    now = str(datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')
    pd.DataFrame({'Dates':record_dates, 'Operations':record_operations, 'Names':record_names, 'Amounts':record_amounts, 'Prices':record_prices, 'Capitals':record_capitals, 'Transaction Costs':record_transation_costs}).to_pickle(predictions_path + "buy_sell_model_class2_" + now)
    pd.DataFrame.from_dict(record_shares).to_pickle(predictions_path + "buy_sell_model_class2_shares_" + now)
                            
    return capital, shares


def buy_sell_class3(predictions_name, adj_close, initial_capital = 10000.0, transaction_cost = 5, predictions_path = "../result/"):
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
    record_transation_costs = []

    # amount of each share after an operation, thus, have the same amount of rows as the lists above
    record_shares = []

    # start buying and selling
    current_date = datetime.strptime(min_date, '%Y-%m-%d')
    while current_date < datetime.strptime(max_date, '%Y-%m-%d'):
        
        # find higher stocks and their prices on the current date
        higher_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), np.argmax(np.concatenate((predictions['Pred0'].values.reshape(-1,1), predictions['Pred1'].values.reshape(-1,1), predictions['Pred2'].values.reshape(-1,1)), axis = 1), axis = 1) == 2)]
        
        if not higher_stocks.empty:

            for higher_stock in higher_stocks['Name']:
            
                higher_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == higher_stock)]['Adj_Close'].iloc[0])
            
                # if we have enough capital to buy the higher_stock, buy it with all of our money
                if capital > higher_price + transaction_cost:
                
                    print("Stock {} will go up, buying...".format(higher_stock))
                    higher_amount = (capital - transaction_cost) // higher_price
                    shares[higher_stock] += higher_amount
                    capital -= higher_amount * higher_price
                    capital -= transaction_cost  # sub transaction cost                    
                    
                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('buy')
                    record_names.append(higher_stock)
                    record_amounts.append(higher_amount)
                    record_prices.append(higher_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)
        
        lower_stocks = predictions.loc[np.logical_and(predictions['Date'] == current_date.strftime("%Y-%m-%d"), np.argmax(np.concatenate((predictions['Pred0'].values.reshape(-1,1), predictions['Pred1'].values.reshape(-1,1), predictions['Pred2'].values.reshape(-1,1)), axis = 1), axis = 1) == 0)]['Name']
                
        # if lower_stocks is not empty
        if not lower_stocks.empty:

            # for each stock in lower_stocks
            for lower_stock in lower_stocks:

                # if we have any of this stock
                if shares[lower_stock] > 0:
                    
                    # find the price of this stock on the current date
                    lower_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == lower_stock)]['Adj_Close'].iloc[0])

                    # sell this stock
                    print("Stocks {} will go down, selling...".format(lower_stock))
                    lower_amount = shares[lower_stock]
                    capital += lower_price * lower_amount
                    shares[lower_stock] = 0
                    capital -= transaction_cost #sub transaction cost

                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('sell')
                    record_names.append(lower_stock)
                    record_amounts.append(lower_amount)
                    record_prices.append(lower_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)

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
                current_share_price = float(adj_close[np.logical_and(adj_close['Date'] == current_date.strftime("%Y-%m-%d"), adj_close['Name'] == current_share)]['Adj_Close'].iloc[0])
                
                # sell this stock
                current_amount = shares[current_share]
                if current_amount > 0: #if we have this share
                    capital += current_share_price * current_amount
                    shares[current_share] = 0
                    capital -= transaction_cost

                    record_dates.append(current_date.strftime("%Y-%m-%d"))
                    record_operations.append('sell')
                    record_names.append(current_share)
                    record_amounts.append(current_amount)
                    record_prices.append(current_share_price)
                    record_capitals.append(capital)
                    record_shares.append(copy.copy(shares))
                    record_transation_costs.append(transaction_cost)

            print("Date: " + current_date.strftime("%Y-%m-%d"))
            print("Capital: " + str(capital))
            print("Shares: " + str(shares))
            print("----------------------------")

    # save records to file
    now = str(datetime.now())
    now = now.replace('-', '_').replace(':', '_').replace('.', '_')
    pd.DataFrame({'Dates':record_dates, 'Operations':record_operations, 'Names':record_names, 'Amounts':record_amounts, 'Prices':record_prices, 'Capitals':record_capitals, 'Transaction Costs':record_transation_costs}).to_pickle(predictions_path + "buy_sell_model_class3_" + now)
    pd.DataFrame.from_dict(record_shares).to_pickle(predictions_path + "buy_sell_model_class3_shares_" + now)
                            
    return capital, shares

