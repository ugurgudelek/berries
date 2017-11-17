"""This file created with the intention of implementation of new rsi,macd,... metrics.
Difference from regular one is these metrics store previous states and we can feed them one by one."""

import pandas as pd
from utils import Bucket
import random
from collections import defaultdict


class MetricEngine:
    def __init__(self, stock_names):
        self.stock_names = stock_names
        self.metrics = defaultdict(dict)  # self.metrics will store metrics with respect to first their stock_name and then uid.

    def add(self, stock_name, metric):
        self.metrics[stock_name][metric.uid] = metric

    def add_default_metrics(self):
        for stock_name in self.stock_names:
            self.add(stock_name, RSI(15))
            self.add(stock_name, RSI(20))
            self.add(stock_name, RSI(25))
            self.add(stock_name, RSI(30))
            self.add(stock_name, SMA(15))
            self.add(stock_name, SMA(20))
            self.add(stock_name, SMA(25))
            self.add(stock_name, SMA(30))
            self.add(stock_name, MACD(26, 12))
            self.add(stock_name, MACD(28, 14))
            self.add(stock_name, MACD(30, 16))
            self.add(stock_name, MACD_Trigger(9, 26, 12))
            self.add(stock_name, MACD_Trigger(10, 28, 14))
            self.add(stock_name, MACD_Trigger(11, 30, 16))
            self.add(stock_name, WilliamR(14))
            self.add(stock_name, WilliamR(18))
            self.add(stock_name, WilliamR(22))
            self.add(stock_name, KDDiff(14))
            self.add(stock_name, KDDiff(18))
            self.add(stock_name, KDDiff(22))
            self.add(stock_name, UltimateOscillator(7, 14, 28))
            self.add(stock_name, UltimateOscillator(8, 16, 32))
            self.add(stock_name, UltimateOscillator(9, 18, 36))
            self.add(stock_name, MoneyFlowIndex(14))
            self.add(stock_name, MoneyFlowIndex(18))
            self.add(stock_name, MoneyFlowIndex(22))

    def feed(self, row):
        """row should be a dict and should have 'stock_name','date' and 'data' keys
        row = dict('stock_name','date','data')
        """
        stock_name = row['stock_name']
        data = row['data']
        date = row['date']

        calculation = pd.Series()
        for (uid, metric) in self.metrics[stock_name].items():
            c = metric.feed(row)
            calculation[uid] = c

        if self.is_proper(calculation):
            return calculation

        return None

    def is_proper(self, metric_data):
        return not any(pd.isnull(metric_data)) # if metric_data has not any None


class Metric:
    def __init__(self):
        pass


class RSI:
    """Relative Strength Index (RSI)
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi"""

    def __init__(self, period):
        self.period = period
        self.uid = 'rsi_' + str(self.period)

        self.average_gain = 0.0
        self.average_loss = 0.0

        self.RS = 0
        self.RSI = 0

        self.gains = Bucket(size=period)
        self.losses = Bucket(size=period)

        # storages
        self.data = []
        self.average_gains = []
        self.average_losses = []
        self.RSs = []
        self.RSIs = []

    def __len__(self):
        return len(self.data)

    def last_data(self):
        return self.data[-1]

    def feed(self, row):
        """row should be a dict and should have 'date' and 'data' keys
        row = dict('date','data')
        """
        data = row['data']
        date = row['date']

        # calculate gain and loss
        if self.__len__() == 0:  # first data
            diff = 0.0
        else:
            diff = data - self.last_data()

        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)

        # put gain and loss into bucket
        self.gains.put(gain)
        self.losses.put(loss)

        # calculate average gain and loss
        if self.__len__() <= self.period:  # first few iteration
            self.average_gain = self.gains.average()
            self.average_loss = self.losses.average()
            self.RS = None
            self.RSI = None
        else:
            self.average_gain = (self.average_gain * (self.period - 1) + gain) / self.period
            self.average_loss = (self.average_loss * (self.period - 1) + loss) / self.period
            # calculate RS and RSI
            try:
                self.RS = self.average_gain / self.average_loss
                self.RSI = 100 - (100 / (1 + self.RS))
            except ZeroDivisionError:  # whenever average_loss is 0 set rsi to 100
                self.RS = 9999
                self.RSI = 100

        # store attributes
        self.data.append(data)
        self.average_gains.append(self.average_gain)
        self.average_losses.append(self.average_loss)
        self.RSIs.append(self.RSI)
        self.RSs.append(self.RS)

        return self.RSI

    def __str__(self):
        string = ""
        for (key, item) in self.__dict__.items():
            string += str(key) + ":" + str(item) + "\n"
        return string

    def __repr__(self):
        return self.__str__()

    def containers_to_dataframe(self):
        df = pd.DataFrame()
        for (key, item) in self.__dict__.items():
            if type(item) is list:
                df[key] = item

        return df


class EMA:
    # todo: Implementation
    def __init__(self, period):
        self.period = period
        self.uid = 'ema_' + str(self.period)

    def feed(self, row):
        return random.random()


class MACD:
    # todo: Implementation
    def __init__(self, period_long=26, period_short=12):
        self.period_long = period_long
        self.period_short = period_short
        self.uid = 'macd_' + str(self.period_long) + '_' + str(self.period_short)

    def feed(self, row):
        return random.random()


class MACD_Trigger:
    # todo: Implementation
    def __init__(self, period_signal=9, period_long=26, period_short=12):
        self.period_signal = period_signal
        self.period_long = period_long
        self.period_short = period_short
        self.uid = 'macd_trig' \
                   + '_' + str(self.period_signal) \
                   + '_' + str(self.period_long) \
                   + '_' + str(self.period_short)

    def feed(self, row):
        return random.random()


class SMA:
    # todo: Implementation
    def __init__(self, period):
        self.period = period
        self.uid = 'sma_' + str(self.period)

    def feed(self, row):
        return random.random()


class WilliamR:
    # todo: Implementation
    def __init__(self, period):
        self.period = period
        self.uid = 'wR_' + str(self.period)

    def feed(self, row):
        return random.random()


class KDDiff:
    # todo: Implementation
    def __init__(self, period):
        self.period = period
        self.uid = 'kddiff_' + str(self.period)

    def feed(self, row):
        return random.random()


class UltimateOscillator:
    # todo: Implementation
    def __init__(self, period1=7, period2=14, period3=28):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.uid = 'macd_trig' \
                   + '_' + str(self.period1) \
                   + '_' + str(self.period2) \
                   + '_' + str(self.period3)

    def feed(self, row):
        return random.random()


class MoneyFlowIndex:
    # todo: Implementation
    def __init__(self, period):
        self.period = period
        self.uid = 'mfi_' + str(self.period)

    def feed(self, row):
        return random.random()
