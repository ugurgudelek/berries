import pandas as pd
import os
import numpy as np

def merge_stocks(input_path, output_path):
    filelist = os.listdir(input_path)
    stocks = dict()

    for file in filelist:
        stockname, extension = os.path.splitext(file)
        if extension == '.csv':
            stockdata = pd.read_csv(os.path.join(input_path, file))
            stockdata['name'] = stockname
            stocks[stockname] = stockdata

    stock_dataframe = pd.DataFrame()
    for stockname, stockdata in stocks.items():
        stock_dataframe = pd.concat((stock_dataframe, stockdata))

    stock_dataframe.to_csv(os.path.join(output_path, 'stocks.csv'), index=False)

def roll_is_max(x):
    return True if x.max() == x[0] else False

def roll_is_min(x):
    return True if x.min() == x[0] else False




def pick_random_samples(df, on, condition, n):
    return df.loc[df[on] == condition].sample(n=n, replace=True)

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


# save some column
def save_column(stocks, col_name='adjusted_close'):
    def inner_func(data):
        data['raw_{}'.format(col_name)] = data[col_name].values
        return data

    return stocks.groupby('name').apply(inner_func)


input_path = '../dataset/finance/stocks/raw_stocks'
output_path = '../dataset/finance/stocks'
merge_stocks(input_path=input_path, output_path=output_path)