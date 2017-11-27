"""This file created with the intention of implementation of new rsi,macd,... metrics.
Difference from regular one is these metrics store previous states and we can feed them one by one."""

import pandas as pd
from utils import Bucket
import random
from collections import defaultdict


class MetricEngine:
    def __init__(self, stock_names):
        self.stock_names = stock_names
        self.metrics = defaultdict(
            dict)  # self.metrics will store metrics with respect to first their stock_name and then uid.

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
        """row should be a dict and should have 'stock_name','date' and 'close' keys
        row = dict('stock_name','date','close')
        """
        stock_name = row['stock_name']
        data = row['close']
        date = row['date']

        calculation = pd.Series()
        for (uid, metric) in self.metrics[stock_name].items():
            c = metric.feed(row)
            calculation[uid] = c

        if self.is_proper(calculation):
            return calculation

        return None

    def is_proper(self, metric_data):
        return not any(pd.isnull(metric_data))  # if metric_data has not any None


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

        self.data = []

        # storages
        self.average_gains = []
        self.average_losses = []
        self.RSs = []
        self.RSIs = []

    def __len__(self):
        return len(self.data)

    def last_data(self):
        return self.data[-1]

    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
        row = dict('date','close')
        """
        data = row['close']
        date = row['date']

        # calculate gain and loss
        if self.__len__() == 0:  # first data
            diff = 0.0
        else:
            diff = data - self.last_data()

        gain = max(diff, 0.0)
        loss = max(-diff, 0.0)

        self.data.append(data)
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
    """http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages"""

    def __init__(self, period):
        self.period = period
        self.uid = 'ema_' + str(self.period)

        self.sma = SMA(period=period)
        self.smoothing_constant = 2 / (self.period + 1)
        self.previous_ema = None

        self.EMA = None
        self.data = []

        self.EMAs = []


    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
                row = dict('date','close')
                """
        data = row['close']
        date = row['date']

        self.data.append(data)
        sma = self.sma.feed(row=row)

        if sma is None:
            self.EMA = None
        else:
            if self.previous_ema is None:
                self.EMA = sma
            else:
                self.EMA = self.smoothing_constant * (data - self.previous_ema) + self.previous_ema

        # store current ema for next calculation
        self.previous_ema = self.EMA
        self.EMAs.append(self.EMA)

        return self.EMA


class MACD:
    """Tested:
    http://investexcel.net/how-to-calculate-macd-in-excel/"""
    def __init__(self, period_long=26, period_short=12):
        if period_long <= period_short:
            raise ValueError("period_long should be bigger than period_short")

        self.period_long = period_long
        self.period_short = period_short
        self.uid = 'macd_' + str(self.period_long) + '_' + str(self.period_short)

        self.ema_long = EMA(period=self.period_long)
        self.ema_short = EMA(period=self.period_short)

        self.MACD = None
        self.data = []
        self.MACDs = []

    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
                        row = dict('date','close')
                        """
        data = row['close']
        date = row['date']
        self.data.append(data)

        ema_long = self.ema_long.feed(row=row)
        ema_short = self.ema_short.feed(row=row)

        if ema_long is None or ema_short is None:
            self.MACD = None
        else:
            self.MACD = ema_short - ema_long

        self.MACDs.append(self.MACD)

        return self.MACD


class MACD_Trigger:
    """Tested:
    http://investexcel.net/how-to-calculate-macd-in-excel/"""
    def __init__(self, period_signal=9, period_long=26, period_short=12):
        self.period_signal = period_signal
        self.period_long = period_long
        self.period_short = period_short
        self.uid = 'macd_trig' \
                   + '_' + str(self.period_signal) \
                   + '_' + str(self.period_long) \
                   + '_' + str(self.period_short)

        self.macd_line = MACD(period_long, period_short)
        self.signal_line = EMA(period_signal)

        self.MACD_HISTOGRAM = None
        self.data = []

        self.MACD_HISTOGRAMs = []

    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
                                row = dict('date','close')
                                """
        data = row['close']
        date = row['date']
        self.data.append(data)

        macd_line = self.macd_line.feed(row=row)
        if macd_line is not None:
            signal_line = self.signal_line.feed(row={'date': date, 'close': macd_line})
            if signal_line is not None:
                self.MACD_HISTOGRAM = macd_line - signal_line
        else:
            self.MACD_HISTOGRAM = None

        self.MACD_HISTOGRAMs.append(self.MACD_HISTOGRAM)

        return self.MACD_HISTOGRAM


class SMA:
    def __init__(self, period):
        self.period = period
        self.uid = 'sma_' + str(self.period)

        self.sma_bucket = Bucket(size=period)
        self.data = []
        self.SMA = None

        self.SMAs = []

    def __len__(self):
        return len(self.data)

    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
                row = dict('date','close')
                """
        data = row['close']
        date = row['date']

        self.data.append(data)

        self.sma_bucket.put(data=data)

        # calculate gain and loss
        if self.sma_bucket.full():  # first few iteration
            self.SMA = self.sma_bucket.average()
        else:
            self.SMA = None

        self.SMAs.append(self.SMA)
        return self.SMA


class WilliamR:
    """Calculates the Williams %R indicator.
    Tested: previous data"""
    def __init__(self, period):
        self.period = period
        self.uid = 'wR_' + str(self.period)

        self.low_bucket = Bucket(size=self.period)
        self.high_bucket = Bucket(size=self.period)

        self.R = None
        self.Rs = []
        self.row_contaioner = []

    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
                        row = dict('date','high','low','close')
                        """

        date = row['date']
        high = row['high']
        low = row['low']
        close = row['close']

        self.row_contaioner.append(row)

        self.low_bucket.put(data=low)
        self.high_bucket.put(data=high)

        if self.low_bucket.full():  # we can calculate now
            highest_high = self.high_bucket.max()
            lowest_low = self.low_bucket.min()
            self.R = (highest_high - close) / (highest_high - lowest_low)*(-100.0)
        else:
            self.R = None

        self.Rs.append(self.R)

        return self.R


    # "Calculates the Williams %R indicator."
    #
    # result = []
    #
    # for curInd in range(period_in_days - 1, data['adjusted_close'].shape[0]):
    #     # current close
    #     curClose = data['close'].values[curInd]
    #
    #     # find the highest high and the lowest low in the period
    #     highestHigh = np.amax(data['high'].values[curInd - period_in_days + 1: curInd + 1])
    #     lowestLow = np.amin(data['low'].values[curInd - period_in_days + 1: curInd + 1])
    #
    #     # calculate %R
    #     wR = (highestHigh - curClose) / (highestHigh - lowestLow) * (-100)
    #     result.append(wR)
    #
    # return np.asarray(result)


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


if __name__ == "__main__":
    ema = MACD_Trigger(period_long=26, period_short=12, period_signal=9)
    ema.feed({'date': '19.02.2013', 'close': 459.99})
    ema.feed({'date': '20.02.2013', 'close': 448.85})
    ema.feed({'date': '21.02.2013', 'close': 446.06})
    ema.feed({'date': '22.02.2013', 'close': 450.81})
    ema.feed({'date': '25.02.2013', 'close': 442.8})
    ema.feed({'date': '26.02.2013', 'close': 448.97})
    ema.feed({'date': '27.02.2013', 'close': 444.57})
    ema.feed({'date': '28.02.2013', 'close': 441.4})
    ema.feed({'date': '1.03.2013', 'close': 430.47})
    ema.feed({'date': '4.03.2013', 'close': 420.05})
    ema.feed({'date': '5.03.2013', 'close': 431.14})
    ema.feed({'date': '6.03.2013', 'close': 425.66})
    ema.feed({'date': '7.03.2013', 'close': 430.58})
    ema.feed({'date': '8.03.2013', 'close': 431.72})
    ema.feed({'date': '11.03.2013', 'close': 437.87})
    ema.feed({'date': '12.03.2013', 'close': 428.43})
    ema.feed({'date': '13.03.2013', 'close': 428.35})
    ema.feed({'date': '14.03.2013', 'close': 432.5})
    ema.feed({'date': '15.03.2013', 'close': 443.66})
    ema.feed({'date': '18.03.2013', 'close': 455.72})
    ema.feed({'date': '19.03.2013', 'close': 454.49})
    ema.feed({'date': '20.03.2013', 'close': 452.08})
    ema.feed({'date': '21.03.2013', 'close': 452.73})
    ema.feed({'date': '22.03.2013', 'close': 461.91})
    ema.feed({'date': '25.03.2013', 'close': 463.58})
    ema.feed({'date': '26.03.2013', 'close': 461.14})
    ema.feed({'date': '27.03.2013', 'close': 452.08})
    ema.feed({'date': '28.03.2013', 'close': 442.66})
    ema.feed({'date': '1.04.2013', 'close': 428.91})
    ema.feed({'date': '2.04.2013', 'close': 429.79})
    ema.feed({'date': '3.04.2013', 'close': 431.99})
    ema.feed({'date': '4.04.2013', 'close': 427.72})
    ema.feed({'date': '5.04.2013', 'close': 423.2})
    ema.feed({'date': '8.04.2013', 'close': 426.21})
    ema.feed({'date': '9.04.2013', 'close': 426.98})
    ema.feed({'date': '10.04.2013', 'close': 435.69})
    ema.feed({'date': '11.04.2013', 'close': 434.33})
    ema.feed({'date': '12.04.2013', 'close': 429.8})
    ema.feed({'date': '15.04.2013', 'close': 419.85})
    ema.feed({'date': '16.04.2013', 'close': 426.24})
    ema.feed({'date': '17.04.2013', 'close': 402.8})
    ema.feed({'date': '18.04.2013', 'close': 392.05})
    ema.feed({'date': '19.04.2013', 'close': 390.53})
    ema.feed({'date': '22.04.2013', 'close': 398.67})
    ema.feed({'date': '23.04.2013', 'close': 406.13})
    ema.feed({'date': '24.04.2013', 'close': 405.46})
    ema.feed({'date': '25.04.2013', 'close': 408.38})
    ema.feed({'date': '26.04.2013', 'close': 417.2})
    ema.feed({'date': '29.04.2013', 'close': 430.12})
    ema.feed({'date': '30.04.2013', 'close': 442.78})
    ema.feed({'date': '1.05.2013', 'close': 439.29})
    ema.feed({'date': '2.05.2013', 'close': 445.52})
    ema.feed({'date': '3.05.2013', 'close': 449.98})
    ema.feed({'date': '6.05.2013', 'close': 460.71})
    ema.feed({'date': '7.05.2013', 'close': 458.66})
    ema.feed({'date': '8.05.2013', 'close': 463.84})
    ema.feed({'date': '9.05.2013', 'close': 456.77})
    ema.feed({'date': '10.05.2013', 'close': 452.97})
    ema.feed({'date': '13.05.2013', 'close': 454.74})
    ema.feed({'date': '14.05.2013', 'close': 443.86})
    ema.feed({'date': '15.05.2013', 'close': 428.85})
    ema.feed({'date': '16.05.2013', 'close': 434.58})
    ema.feed({'date': '17.05.2013', 'close': 433.26})
    ema.feed({'date': '20.05.2013', 'close': 442.93})
    ema.feed({'date': '21.05.2013', 'close': 439.66})
    ema.feed({'date': '22.05.2013', 'close': 441.35})
