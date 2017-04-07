import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import metrics as mt
import yahoo_finance_io
import datetime
from dateutil import parser
import clustering


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


# def calculate_metrics(data, metric_functions):
#     # lets calculate some metric over dataset
#     ret = []
#     for func in metric_functions:
#         ret.append(func(data))

#     # rsi_15_data = mt.rsi(data)
#     # sma_15_data = mt.sma(data)
#     # macd_15_5_data = mt.macd(data)
#     # macd_trigger_9_15_5 = mt.macd_trigger(data)
#     # willR = mt.williamsR(data)
#     # kdHist = mt.kdDiff(data)
#     # ultimateOs = mt.ulOs(data)
#     # mfIndex = mt.mfi(data)

#     return np.asarray(ret)

def calculate_metrics(data, metric_functions):

    ret = []
    for func in metric_functions:
        ret.append(func(data))

    rsi_15_data = mt.rsi(data, 15); rsi_20_data = mt.rsi(data, 20); rsi_25_data = mt.rsi(data, 25); rsi_30_data = mt.rsi(data, 30)
    rsi_35_data = mt.rsi(data, 35); rsi_40_data = mt.rsi(data, 40); rsi_45_data = mt.rsi(data, 45); rsi_50_data = mt.rsi(data, 50)
    
    sma_15_data = mt.sma(data, 15); sma_20_data = mt.sma(data, 20); sma_25_data = mt.sma(data, 25); sma_30_data = mt.sma(data, 30)
    sma_35_data = mt.sma(data, 35); sma_40_data = mt.sma(data, 40); sma_45_data = mt.sma(data, 45); sma_50_data = mt.sma(data, 50)
    
    macd_26_12_data = mt.macd(data, 26, 12); macd_28_14_data = mt.macd(data, 28, 14); macd_30_16_data = mt.macd(data, 30, 16); macd_32_18_data = mt.macd(data, 32, 18)
    macd_32_20_data = mt.macd(data, 32, 20); macd_34_22_data = mt.macd(data, 34, 22); macd_36_24_data = mt.macd(data, 36, 24); macd_38_26_data = mt.macd(data, 38, 26)
    
    macd_trigger_9_26_12 = mt.macd_trigger(data, 9, 26, 12); macd_trigger_10_28_14 = mt.macd_trigger(data, 10, 28, 14); macd_trigger_11_30_16 = mt.macd_trigger(data, 11, 30, 16); macd_trigger_12_32_18 = mt.macd_trigger(data, 12, 32, 18)
    macd_trigger_13_34_20 = mt.macd_trigger(data, 13, 34, 20); macd_trigger_14_36_22 = mt.macd_trigger(data, 14, 36, 22); macd_trigger_15_38_24 = mt.macd_trigger(data, 15, 38, 24); macd_trigger_16_40_26 = mt.macd_trigger(data, 16, 40, 26)
    
    willR_14 = mt.williamsR(data, 14); willR_18 = mt.williamsR(data, 18); willR_22 = mt.williamsR(data, 22); willR_26 = mt.williamsR(data, 26)
    willR_30 = mt.williamsR(data, 30); willR_34 = mt.williamsR(data, 34); willR_38 = mt.williamsR(data, 38); willR_42 = mt.williamsR(data, 42)
    
    kdHist_14 = mt.kdDiff(data, 14); kdHist_18 = mt.kdDiff(data, 18); kdHist_22 = mt.kdDiff(data, 22); kdHist_26 = mt.kdDiff(data, 26)
    kdHist_30 = mt.kdDiff(data, 30); kdHist_34 = mt.kdDiff(data, 34); kdHist_38 = mt.kdDiff(data, 38); kdHist_42 = mt.kdDiff(data, 42)
    
    ultimateOs_7_14_28 = mt.ulOs(data, 7, 14, 28); ultimateOs_8_16_32 = mt.ulOs(data, 8, 16, 32); ultimateOs_9_18_36 = mt.ulOs(data, 9, 18, 36); ultimateOs_10_20_40 = mt.ulOs(data, 10, 20, 40)
    ultimateOs_11_22_44 = mt.ulOs(data, 11, 22, 44); ultimateOs_12_24_48 = mt.ulOs(data, 12, 24, 48); ultimateOs_13_26_52 = mt.ulOs(data, 13, 26, 52); ultimateOs_14_28_56 = mt.ulOs(data, 14, 28, 56)
    
    mfIndex_14 = mt.mfi(data, 14); mfIndex_18 = mt.mfi(data, 18); mfIndex_22 = mt.mfi(data, 22); mfIndex_26 = mt.mfi(data, 26)
    mfIndex_30 = mt.mfi(data, 30); mfIndex_34 = mt.mfi(data, 34); mfIndex_38 = mt.mfi(data, 38); mfIndex_40 = mt.mfi(data, 40)

    return np.asarray(ret)    


def stack_data_and_metrics(data, metrics, metric_functions):
    for i, metric_data in enumerate(metrics):
        data[metric_functions[i].__name__] = metric_data
    return data


def tranform_to_dict(data):
    # create a dict for easy use
    data_dict = {'date': data[:, 0],
                 'open': data[:, 1],
                 'high': data[:, 2],
                 'low': data[:, 3],
                 'close': data[:, 4],
                 'volume': data[:, 5],
                 'adjusted_close': data[:, 6]}
    return data_dict


def data_handler(which_stock, start_date, end_date, metric_functions, is_save_csv=True):
    # Open High Low Close Volume Adj Close
    stock = yahoo_finance_io.data_getter(which_stock, start_date, end_date)
    if stock is not None:
        stock = stock.as_matrix()[:, :][::-1]

        # Tranform array to dict for easy use
        stock = tranform_to_dict(stock)

        # Adjust data according to adjusted close
        stock = adjusted_data(stock)

        # create data arr to hold all metric info
        metrics = calculate_metrics(stock)

        # assign nan value beginning of the data
        metrics = assign_null_into_data(arr=metrics, length=len(stock['adjusted_close']))

        # append data and metrics column-wise
        stock = stack_data_and_metrics(stock, metrics, metric_functions)

        df = pd.DataFrame(stock)

        if is_save_csv:
            df.to_csv("data/" + which_stock + ".csv")

        return df


def get_features(which_stock):
    """takes stock names and returns proper feature set"""
    df = pd.read_csv("data/spy.csv")
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('date', axis=1)
    df = df.dropna()

    features_df = df.drop(['low', 'close', 'high', 'open'], axis=1)
    return features_df



def main():
    # stock_names = ['spy', 'gdx', 'xlf', 'jnug', 'eem', 'nugt', 'vxx', 'iwm', 'gdxj', 'uso', 'efa', 'uvxy', 'qqq', 'fxi',
    #                'jdst', 'ewz', 'xlu', 'xle', 'ung', 'xiv', 'xop', 'vwo', 'xlp', 'hyg', 'jnk', 'xli', 'tlt', 'tza',
    #                'xlv', 'rsx', 'ugaz', 'amlp', 'dust', 'vea', 'iemg', 'uco', 'xlk', 'iau', 'gld', 'kre', 'sds', 'iyr',
    #                'xrt', 'slv', 'dgaz', 'ewj', 'xbi', 'oih', 'ezu', 'xlb', 'lqd', 'bkln', 'vnq', 'ijh', 'labd', 'xly',
    #                'iefa', 'spxu', 'xme', 'ewt', 'dxj', 'eww', 'spxs', 'dia', 'fas', 'ivv', 'ijr', 'tna', 'inda', 'ewg',
    #                'pff', 'vgk', 'svxy', 'agg', 'ewh', 'dbc', 'ewc', 'iwf', 'epi', 'kbe', 'sso', 'vixy', 'oil', 'labu',
    #                'tbt', 'sqqq', 'itb', 'tqqq', 'ewu', 'bnd', 'ewa', 'vti', 'voo', 'fez', 'emb', 'iwd', 'uup', 'ewy',
    #                'fxn', 'xlre']

    # stock_names = ['spy', 'xlf' , 'qqq' , 'xlu' , 'xle' , 'xlp' , 'xli' , 'xlv' , 'xlk' , 'ewj' , 'xlb', 'xly', 'eww',
    #                'dia', 'ewg', 'ewh', 'ewc', 'ewu','ewa']

    # functions to pass other methods
    metric_functions = [mt.rsi, mt.sma, mt.macd, mt.macd_trigger, mt.williamsR, mt.kdDiff, mt.ulOs, mt.mfi]

    # example of yahoo finance data getter function
    start_date = datetime.date(2000, 1, 3)
    end_date = datetime.date(2017, 1, 1)

    # gaugeCounter = 1
    # available_etfs = []
    # for stock in stock_names:
    #     print(gaugeCounter)
    #     df = data_handler(stock, start_date, end_date, metric_functions, is_save_csv=True)
    #     if df is not None:
    #         available_etfs.append(stock)
    #         print(stock)
    #     gaugeCounter += 1

    # start_dates = []
    # years = []
    # for etf in stock_names:
    #     df = pd.read_csv("data/"+etf+".csv")
    #     # print(df.date.iloc[0])
    #     start_dates.append(df.date.iloc[0])
    #     years.append(parser.parse(df.date.iloc[0]).year)


    # sort features according to pearson correlation coefficient
    features_df = get_features('spy')
    sorted_cluster_names = clustering.hierarchical_clustering(features_df)

    #
    # # plt.plot(sma_15_data, color='r')
    # # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # # plt.show()


if __name__ == "__main__":
    main()
