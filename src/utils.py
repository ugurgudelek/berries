import pandas as pd
import os

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

    stock_dataframe.to_csv(os.path.join(output_path, 'stocks.csv'))


path = '../dataset/finance/stocks'
merge_stocks(input_path=path, output_path=path)