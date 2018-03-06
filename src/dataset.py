"""
Ugur Gudelek
dataset
ugurgudelek
06-Mar-18
finance-cnn
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from talib import RSI, SMA, MACD, WILLR, ULTOSC, MFI, STOCH

import os




class Config:
    """

    """
    def __init__(self):
        self.stocks_dir = '../input/raw_data'

class IndicatorDataset(Dataset):
    """

    """

    def __init__(self, config, stock_names=None):
        self.stocks_dir = config.stocks_dir

        self.stocks = self._read_dir(self.stocks_dir)

        # if stock_names are assigned, drop some stock
        if stock_names is not None:
            keys = list(self.stocks.keys())
            for stock_name in keys:
                if stock_name not in stock_names:
                    self.stocks.pop(stock_name)

        self.dataset = IndicatorDataset.stocks_to_dataset(self.stocks)



    def _read_dir(self, stocks_dir):
        stocks = dict()
        for fullfilename in os.listdir(stocks_dir):
            filename, extension = fullfilename.split('.')
            if extension == 'csv':  # check extension
                stocks[filename] = pd.read_csv(os.path.join(stocks_dir, fullfilename))

        return stocks

    def __len__(self):
        pass

    def __getitem__(self, ix):
        pass


    @staticmethod
    def stocks_to_dataset(stocks):
        """

        Args:
            stocks:

        Returns:

        """
        r = dict()
        for (name, stock) in stocks.items():
            r[name] = IndicatorDataset.indicators(stock.copy())
        return r

    @staticmethod
    def indicators(dataframe: pd.DataFrame) -> pd.DataFrame:
        """

        Args:
            dataframe(pd.DataFrame): should have 'high', 'low', 'adjusted_close', 'volume'

        Returns: (pd.DataFrame) calculated indicators

        """

        high = dataframe['high'].values.astype(np.float)
        low = dataframe['low'].values.astype(np.float)
        close = dataframe['adjusted_close'].values.astype(np.float)
        volume = dataframe['volume'].values.astype(np.float)

        dataframe['rsi_15'] = RSI(close, timeperiod=15)
        dataframe['rsi_20'] = RSI(close, timeperiod=20)
        dataframe['rsi_25'] = RSI(close, timeperiod=25)
        dataframe['rsi_30'] = RSI(close, timeperiod=30)

        dataframe['sma_15'] = SMA(close, timeperiod=15)
        dataframe['sma_20'] = SMA(close, timeperiod=20)
        dataframe['sma_25'] = SMA(close, timeperiod=25)
        dataframe['sma_30'] = SMA(close, timeperiod=30)

        dataframe['macd_12'], macdsignal, dataframe['macdhist_12'] = MACD(close, fastperiod=12, slowperiod=26,
                                                                          signalperiod=9)
        dataframe['macd_14'], macdsignal, dataframe['macdhist_14'] = MACD(close, fastperiod=14, slowperiod=28,
                                                                          signalperiod=10)
        dataframe['macd_16'], macdsignal, dataframe['macdhist_16'] = MACD(close, fastperiod=16, slowperiod=30,
                                                                          signalperiod=11)

        dataframe['willR_14'] = WILLR(high, low, close, timeperiod=14)
        dataframe['willR_18'] = WILLR(high, low, close, timeperiod=18)
        dataframe['willR_22'] = WILLR(high, low, close, timeperiod=22)

        dataframe['ultimate_osc_7'] = ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        dataframe['ultimate_osc_8'] = ULTOSC(high, low, close, timeperiod1=8, timeperiod2=16, timeperiod3=32)
        dataframe['ultimate_osc_9'] = ULTOSC(high, low, close, timeperiod1=9, timeperiod2=18, timeperiod3=36)

        dataframe['mfi_14'] = MFI(high, low, close, volume, timeperiod=14)
        dataframe['mfi_18'] = MFI(high, low, close, volume, timeperiod=18)
        dataframe['mfi_22'] = MFI(high, low, close, volume, timeperiod=22)

        slowk, slowd = STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_14'] = slowk - slowd
        slowk, slowd = STOCH(high, low, close, fastk_period=18, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_18'] = slowk - slowd
        slowk, slowd = STOCH(high, low, close, fastk_period=22, slowk_period=3, slowk_matype=0, slowd_period=3,
                             slowd_matype=0)
        dataframe['kddiff_22'] = slowk - slowd

        return dataframe




if __name__ == "__main__":
    config = Config()

    dataset = IndicatorDataset(config=config, stock_names=['spy'])

    print(dataset.stocks.keys())
    print(dataset.stocks)
