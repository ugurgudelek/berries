import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import metrics as mt
import yahoo_finance_io
import datetime


def crop_data(arr):
    """crop the non-adjusted data starting from the beginning"""
    min_len = len(arr[0])
    for data in arr[1:]:
        min_len = min(min_len, len(data))

    for i in range(len(arr)):
        # crop beginning of data
        arr[i] = arr[i][len(arr[i]) - min_len:]

    return arr


def assign_null_into_data(arr, length):
    for i, data in enumerate(arr):
        null_len = length - len(data)
        null_part = np.zeros(null_len)
        null_part.fill(np.nan)

        arr[i] = np.hstack((null_part, data))
    return arr


def adjusted_data(data):
    # adjust the prices according to adjusted close
    adj_ratio = data['adjusted_close'] / data['close']
    data['open'] = adj_ratio * data['open']  # open
    data['high'] = adj_ratio * data['high']  # high
    data['low'] = adj_ratio * data['low']  # low
    data['close'] = data['adjusted_close']  # close is adjusted

    return data


def calculate_metrics(data, metric_functions):
    # lets calculate some metric over dataset
    ret = []
    for func in metric_functions:
        ret.append(func(data))

    # rsi_15_data = mt.rsi(data)
    # sma_15_data = mt.sma(data)
    # macd_15_5_data = mt.macd(data)
    # macd_trigger_9_15_5 = mt.macd_trigger(data)
    # willR = mt.williamsR(data)
    # kdHist = mt.kdDiff(data)
    # ultimateOs = mt.ulOs(data)
    # mfIndex = mt.mfi(data)

    return np.asarray(ret)


def stack_data_and_metrics(data, metrics, metric_functions):
    for i, metric_data in enumerate(metrics):
        data[metric_functions[i].__name__] = metric_data
    return data

def tranform_to_dict(data):
    # create a dict for easy use
    data_dict = {'open': data[:, 0],
                 'high': data[:, 1],
                 'low': data[:, 2],
                 'close': data[:, 3],
                 'volume': data[:, 4],
                 'adjusted_close': data[:, 5]}
    return data_dict

def data_handler(which_stock, start_date, end_date, metric_functions, is_save_csv=True):
    # Open High Low Close Volume Adj Close
    stock = yahoo_finance_io.data_getter(which_stock, start_date, end_date).as_matrix()[:, :][::-1]

    # Tranform array to dict for easy use
    stock = tranform_to_dict(stock)

    # Adjust data according to adjusted close
    stock = adjusted_data(stock)

    # create data arr to hold all metric info
    metrics = calculate_metrics(stock, metric_functions)

    # assign nan value beginning of the data
    metrics = assign_null_into_data(arr=metrics, length=len(stock['adjusted_close']))

    # append data and metrics column-wise
    stock = stack_data_and_metrics(stock, metrics, metric_functions)

    df = pd.DataFrame(stock)
    
    if is_save_csv:
        df.to_csv("data/"+which_stock+".csv")
        
    return df




def main():
    
    # functions to pass other methods
    metric_functions = [mt.rsi, mt.sma, mt.macd, mt.macd_trigger, mt.williamsR, mt.kdDiff, mt.ulOs, mt.mfi]

    # example of yahoo finance data getter function
    start_date = datetime.datetime(2000, 1, 3)
    end_date = datetime.datetime(2016, 12, 30)
    
    stock_names = ['spy', 'msft', 'aapl']

    for stock in stock_names:
        df = data_handler(stock, start_date, end_date, metric_functions, is_save_csv=True)
        print(df.tail())



    df = pd.read_csv("data/spy.csv")
    print(df.head())
    print(df.tail())

    #
    # # plt.plot(sma_15_data, color='r')
    # # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # # plt.show()


if __name__ == "__main__":
    main()
