"""This file created with the intention of implementation of new rsi,macd,... metrics.
Difference from regular one is these metrics store previous states and we can feed them one by one."""

import pandas as pd
from utils import Bucket
import random
from collections import defaultdict
import pickle

class MetricEngine:
    def __init__(self, stock_names):
        self.stock_names = stock_names
        self.metrics = defaultdict(
            dict)  # self.metrics will store metrics with respect to first their stock_name and then uid.

    def save_instance(self, filepath, run_number):
        filename = filepath+'/{}_metric_engine.pkl'.format(run_number)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_instance(self, filepath, run_number):
        filename = filepath+'/{}_metric_engine.pkl'.format(run_number)
        with open(filename, 'rb') as f:
            self = pickle.load(f)

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
            self.add(stock_name, KDDifference(14))
            self.add(stock_name, KDDifference(18))
            self.add(stock_name, KDDifference(22))
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
            self.R = (highest_high - close) / (highest_high - lowest_low) * (-100.0)
        else:
            self.R = None

        self.Rs.append(self.R)

        return self.R


class KDDifference:
    """Calculates the difference between %K and %D.
    Tested!
    Calculation:
    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %K is multiplied by 100 to move the decimal point two places"""

    def __init__(self, period, sma_period=3):
        self.period = period
        self.sma_period = sma_period
        self.uid = 'kddiff_' + str(self.period)

        self.low_bucket = Bucket(size=self.period)
        self.high_bucket = Bucket(size=self.period)

        self.sma_3_day = SMA(period=self.sma_period)

        self.K = None
        self.D = None
        self.KD_diff = None
        self.Ks = []
        self.Ds = []
        self.KD_diffs = []

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
            self.K = (close - lowest_low) / (highest_high - lowest_low) * (100.0)
            self.D = self.sma_3_day.feed(row={'date': date, 'close': self.K})
            if self.D is not None:
                self.KD_diff = self.K - self.D
        else:
            self.K = None
            self.D = None
            self.KD_diff = None

        self.Ks.append(self.K)
        self.Ds.append(self.D)
        self.KD_diffs.append(self.KD_diff)

        return self.KD_diff


class UltimateOscillator:
    """Calculates the ultimate oscillator. Periods should be from low to high.
    Tested!
    Calculation:
    BP = Close - Minimum(Low or Prior Close).

    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)

    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)

    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)
"""

    def __init__(self, period1=7, period2=14, period3=28):
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.w1 = self.period3 / self.period1
        self.w2 = self.period3 / self.period2
        self.w3 = self.period3 / self.period3

        self.uid = 'ulos' \
                   + '_' + str(self.period1) \
                   + '_' + str(self.period2) \
                   + '_' + str(self.period3)

        self.buying_pressure = None
        self.true_range = None
        self.prior_close = None

        self.UO = None

        # buying pressure buckets
        self.bp_period1_bucket = Bucket(size=self.period1)
        self.bp_period2_bucket = Bucket(size=self.period2)
        self.bp_period3_bucket = Bucket(size=self.period3)

        # true range buckets
        self.tr_period1_bucket = Bucket(size=self.period1)
        self.tr_period2_bucket = Bucket(size=self.period2)
        self.tr_period3_bucket = Bucket(size=self.period3)

        # storage
        self.row_contaioner = []
        self.UOs = []

    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
                                row = dict('date','high','low','close')
                                """

        date = row['date']
        high = row['high']
        low = row['low']
        close = row['close']

        self.row_contaioner.append(row)
        self.UO = None  # init current UO

        if self.prior_close is not None:

            self.buying_pressure = close - min(low, self.prior_close)  # calculate bp
            self.true_range = max(high, self.prior_close) - min(low, self.prior_close)  # calculate tr

            self.bp_period1_bucket.put(data=self.buying_pressure)
            self.bp_period2_bucket.put(data=self.buying_pressure)
            self.bp_period3_bucket.put(data=self.buying_pressure)
            self.tr_period1_bucket.put(data=self.true_range)
            self.tr_period2_bucket.put(data=self.true_range)
            self.tr_period3_bucket.put(data=self.true_range)

            average_period1 = self.bp_period1_bucket.average() / self.tr_period1_bucket.average() if self.bp_period1_bucket.full() else None
            average_period2 = self.bp_period2_bucket.average() / self.tr_period2_bucket.average() if self.bp_period2_bucket.full() else None
            average_period3 = self.bp_period3_bucket.average() / self.tr_period3_bucket.average() if self.bp_period3_bucket.full() else None

            if (average_period1 is not None) and (average_period2 is not None) and (average_period3 is not None):
                self.UO = 100 * (
                    (self.w1 * average_period1) + (self.w2 * average_period2) + (self.w3 * average_period3)) / (
                              self.w1 + self.w2 + self.w3)

        self.prior_close = close  # set prior_close for next iteration
        self.UOs.append(self.UO)
        return self.UO


class MoneyFlowIndex:
    """Calculates the money flow index for the given period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
    Tested!
    Calculation:
    Typical Price = (High + Low + Close)/3

    Raw Money Flow = Typical Price x Volume
    Money Flow Ratio = (14-period Positive Money Flow)/(14-period Negative Money Flow)

    Money Flow Index = 100 - 100/(1 + Money Flow Ratio)
    """

    def __init__(self, period):
        self.period = period
        self.uid = 'mfi_' + str(self.period)

        self.typical_price = None
        self.raw_money_flow = None
        self.money_flow_ratio = None

        self.prior_typical_price = None

        self.positive_mf_bucket = Bucket(size=self.period)
        self.negative_mf_bucket = Bucket(size=self.period)

        self.MFI = None

        self.row_contaioner = []

        # storage
        self.MFIs = []

    def __len__(self):
        return len(self.row_contaioner)

    def feed(self, row):
        """row should be a dict and should have 'date' and 'close' keys
                                row = dict('date','high','low','close')
                                """

        date = row['date']
        high = row['high']
        low = row['low']
        close = row['close']
        volume = row['volume']

        self.row_contaioner.append(row)

        self.MFI = None  # init current MFI

        self.typical_price = (high + low + close) / 3.0
        self.raw_money_flow = self.typical_price * volume

        if self.prior_typical_price is not None:  # drop first calculation
            if self.typical_price - self.prior_typical_price >= 0:  # if positive money flow
                self.positive_mf_bucket.put(self.raw_money_flow)
                self.negative_mf_bucket.put(0.0)
            else:  # neg money flow
                self.negative_mf_bucket.put(self.raw_money_flow)
                self.positive_mf_bucket.put(0.0)

            if self.__len__() > self.period:
                # we have enough data and can calculate money flow now.
                positive_mf = self.positive_mf_bucket.sum()
                negative_mf = self.negative_mf_bucket.sum()

                # calculate mfr and check zero div error
                self.money_flow_ratio = positive_mf / negative_mf if negative_mf != 0 else 1.0

                self.MFI = 100 - 100 / (1 + self.money_flow_ratio)

        self.prior_typical_price = self.typical_price

        self.MFIs.append(self.MFI)
        return self.MFI


if __name__ == "__main__":
    # ema = MACD_Trigger(period_long=26, period_short=12, period_signal=9)
    # ema.feed({'date': '19.02.2013', 'close': 459.99})
    # ema.feed({'date': '20.02.2013', 'close': 448.85})
    # ema.feed({'date': '21.02.2013', 'close': 446.06})
    # ema.feed({'date': '22.02.2013', 'close': 450.81})
    # ema.feed({'date': '25.02.2013', 'close': 442.8})
    # ema.feed({'date': '26.02.2013', 'close': 448.97})
    # ema.feed({'date': '27.02.2013', 'close': 444.57})
    # ema.feed({'date': '28.02.2013', 'close': 441.4})
    # ema.feed({'date': '1.03.2013', 'close': 430.47})
    # ema.feed({'date': '4.03.2013', 'close': 420.05})
    # ema.feed({'date': '5.03.2013', 'close': 431.14})
    # ema.feed({'date': '6.03.2013', 'close': 425.66})
    # ema.feed({'date': '7.03.2013', 'close': 430.58})
    # ema.feed({'date': '8.03.2013', 'close': 431.72})
    # ema.feed({'date': '11.03.2013', 'close': 437.87})
    # ema.feed({'date': '12.03.2013', 'close': 428.43})
    # ema.feed({'date': '13.03.2013', 'close': 428.35})
    # ema.feed({'date': '14.03.2013', 'close': 432.5})
    # ema.feed({'date': '15.03.2013', 'close': 443.66})
    # ema.feed({'date': '18.03.2013', 'close': 455.72})
    # ema.feed({'date': '19.03.2013', 'close': 454.49})
    # ema.feed({'date': '20.03.2013', 'close': 452.08})
    # ema.feed({'date': '21.03.2013', 'close': 452.73})
    # ema.feed({'date': '22.03.2013', 'close': 461.91})
    # ema.feed({'date': '25.03.2013', 'close': 463.58})
    # ema.feed({'date': '26.03.2013', 'close': 461.14})
    # ema.feed({'date': '27.03.2013', 'close': 452.08})
    # ema.feed({'date': '28.03.2013', 'close': 442.66})
    # ema.feed({'date': '1.04.2013', 'close': 428.91})
    # ema.feed({'date': '2.04.2013', 'close': 429.79})
    # ema.feed({'date': '3.04.2013', 'close': 431.99})
    # ema.feed({'date': '4.04.2013', 'close': 427.72})
    # ema.feed({'date': '5.04.2013', 'close': 423.2})
    # ema.feed({'date': '8.04.2013', 'close': 426.21})
    # ema.feed({'date': '9.04.2013', 'close': 426.98})
    # ema.feed({'date': '10.04.2013', 'close': 435.69})
    # ema.feed({'date': '11.04.2013', 'close': 434.33})
    # ema.feed({'date': '12.04.2013', 'close': 429.8})
    # ema.feed({'date': '15.04.2013', 'close': 419.85})
    # ema.feed({'date': '16.04.2013', 'close': 426.24})
    # ema.feed({'date': '17.04.2013', 'close': 402.8})
    # ema.feed({'date': '18.04.2013', 'close': 392.05})
    # ema.feed({'date': '19.04.2013', 'close': 390.53})
    # ema.feed({'date': '22.04.2013', 'close': 398.67})
    # ema.feed({'date': '23.04.2013', 'close': 406.13})
    # ema.feed({'date': '24.04.2013', 'close': 405.46})
    # ema.feed({'date': '25.04.2013', 'close': 408.38})
    # ema.feed({'date': '26.04.2013', 'close': 417.2})
    # ema.feed({'date': '29.04.2013', 'close': 430.12})
    # ema.feed({'date': '30.04.2013', 'close': 442.78})
    # ema.feed({'date': '1.05.2013', 'close': 439.29})
    # ema.feed({'date': '2.05.2013', 'close': 445.52})
    # ema.feed({'date': '3.05.2013', 'close': 449.98})
    # ema.feed({'date': '6.05.2013', 'close': 460.71})
    # ema.feed({'date': '7.05.2013', 'close': 458.66})
    # ema.feed({'date': '8.05.2013', 'close': 463.84})
    # ema.feed({'date': '9.05.2013', 'close': 456.77})
    # ema.feed({'date': '10.05.2013', 'close': 452.97})
    # ema.feed({'date': '13.05.2013', 'close': 454.74})
    # ema.feed({'date': '14.05.2013', 'close': 443.86})
    # ema.feed({'date': '15.05.2013', 'close': 428.85})
    # ema.feed({'date': '16.05.2013', 'close': 434.58})
    # ema.feed({'date': '17.05.2013', 'close': 433.26})
    # ema.feed({'date': '20.05.2013', 'close': 442.93})
    # ema.feed({'date': '21.05.2013', 'close': 439.66})
    # ema.feed({'date': '22.05.2013', 'close': 441.35})

    # kddiff = KDDifference(period=14)
    # 
    # kddiff.feed({'date': '23-Feb-10', 'high': 127.01, 'low': 125.36, 'close': 127.29})
    # kddiff.feed({'date': '24-Feb-10', 'high': 127.62, 'low': 126.16, 'close': 127.29})
    # kddiff.feed({'date': '25-Feb-10', 'high': 126.59, 'low': 124.93, 'close': 127.29})
    # kddiff.feed({'date': '26-Feb-10', 'high': 127.35, 'low': 126.09, 'close': 127.29})
    # kddiff.feed({'date': '1-Mar-10', 'high': 128.17, 'low': 126.82, 'close': 127.29})
    # kddiff.feed({'date': '2-Mar-10', 'high': 128.43, 'low': 126.48, 'close': 127.29})
    # kddiff.feed({'date': '3-Mar-10', 'high': 127.37, 'low': 126.03, 'close': 127.29})
    # kddiff.feed({'date': '4-Mar-10', 'high': 126.42, 'low': 124.83, 'close': 127.29})
    # kddiff.feed({'date': '5-Mar-10', 'high': 126.90, 'low': 126.39, 'close': 127.29})
    # kddiff.feed({'date': '8-Mar-10', 'high': 126.85, 'low': 125.72, 'close': 127.29})
    # kddiff.feed({'date': '9-Mar-10', 'high': 125.65, 'low': 124.56, 'close': 127.29})
    # kddiff.feed({'date': '10-Mar-10', 'high': 125.72, 'low': 124.57, 'close': 127.29})
    # kddiff.feed({'date': '11-Mar-10', 'high': 127.16, 'low': 125.07, 'close': 127.29})
    # kddiff.feed({'date': '12-Mar-10', 'high': 127.72, 'low': 126.86, 'close': 127.29})
    # kddiff.feed({'date': '15-Mar-10', 'high': 127.69, 'low': 126.63, 'close': 127.18})
    # kddiff.feed({'date': '16-Mar-10', 'high': 128.22, 'low': 126.80, 'close': 128.01})
    # kddiff.feed({'date': '17-Mar-10', 'high': 128.27, 'low': 126.71, 'close': 127.11})
    # kddiff.feed({'date': '18-Mar-10', 'high': 128.09, 'low': 126.80, 'close': 127.73})
    # kddiff.feed({'date': '19-Mar-10', 'high': 128.27, 'low': 126.13, 'close': 127.06})
    # kddiff.feed({'date': '22-Mar-10', 'high': 127.74, 'low': 125.92, 'close': 127.33})
    # kddiff.feed({'date': '23-Mar-10', 'high': 128.77, 'low': 126.99, 'close': 128.71})
    # kddiff.feed({'date': '24-Mar-10', 'high': 129.29, 'low': 127.81, 'close': 127.87})
    # kddiff.feed({'date': '25-Mar-10', 'high': 130.06, 'low': 128.47, 'close': 128.58})
    # kddiff.feed({'date': '26-Mar-10', 'high': 129.12, 'low': 128.06, 'close': 128.60})
    # kddiff.feed({'date': '29-Mar-10', 'high': 129.29, 'low': 127.61, 'close': 127.93})
    # kddiff.feed({'date': '30-Mar-10', 'high': 128.47, 'low': 127.60, 'close': 128.11})
    # kddiff.feed({'date': '31-Mar-10', 'high': 128.09, 'low': 127.00, 'close': 127.60})
    # kddiff.feed({'date': '1-Apr-10', 'high': 128.65, 'low': 126.90, 'close': 127.60})
    # kddiff.feed({'date': '5-Apr-10', 'high': 129.14, 'low': 127.49, 'close': 128.69})

    # uos = UltimateOscillator()
    #
    # uos.feed({'date': '20.Eki.10', 'high': 57.93, 'low': 56.52, 'close': 57.57})
    # uos.feed({'date': '21.Eki.10', 'high': 58.46, 'low': 57.07, 'close': 57.67})
    # uos.feed({'date': '22.Eki.10', 'high': 57.76, 'low': 56.44, 'close': 56.92})
    # uos.feed({'date': '25.Eki.10', 'high': 59.88, 'low': 57.53, 'close': 58.47})
    # uos.feed({'date': '26.Eki.10', 'high': 59.02, 'low': 57.58, 'close': 58.74})
    # uos.feed({'date': '27.Eki.10', 'high': 60.18, 'low': 57.89, 'close': 60.01})
    # uos.feed({'date': '28.Eki.10', 'high': 60.29, 'low': 58.01, 'close': 58.45})
    # uos.feed({'date': '29.Eki.10', 'high': 59.86, 'low': 58.43, 'close': 59.18})
    # uos.feed({'date': '1.Kas.10', 'high': 59.78, 'low': 58.45, 'close': 58.67})
    # uos.feed({'date': '2.Kas.10', 'high': 59.73, 'low': 58.58, 'close': 58.87})
    # uos.feed({'date': '3.Kas.10', 'high': 59.60, 'low': 58.54, 'close': 59.30})
    # uos.feed({'date': '4.Kas.10', 'high': 62.96, 'low': 59.62, 'close': 62.57})
    # uos.feed({'date': '5.Kas.10', 'high': 62.27, 'low': 61.36, 'close': 62.02})
    # uos.feed({'date': '8.Kas.10', 'high': 63.06, 'low': 61.25, 'close': 62.05})
    # uos.feed({'date': '9.Kas.10', 'high': 63.74, 'low': 62.19, 'close': 62.52})
    # uos.feed({'date': '10.Kas.10', 'high': 62.74, 'low': 61.02, 'close': 62.37})
    # uos.feed({'date': '11.Kas.10', 'high': 63.48, 'low': 61.57, 'close': 63.40})
    # uos.feed({'date': '12.Kas.10', 'high': 63.23, 'low': 60.79, 'close': 61.90})
    # uos.feed({'date': '15.Kas.10', 'high': 62.14, 'low': 60.34, 'close': 60.54})
    # uos.feed({'date': '16.Kas.10', 'high': 60.50, 'low': 58.20, 'close': 59.09})
    # uos.feed({'date': '17.Kas.10', 'high': 59.89, 'low': 58.91, 'close': 59.01})
    # uos.feed({'date': '18.Kas.10', 'high': 60.32, 'low': 59.09, 'close': 59.39})
    # uos.feed({'date': '19.Kas.10', 'high': 59.71, 'low': 58.59, 'close': 59.21})
    # uos.feed({'date': '22.Kas.10', 'high': 62.22, 'low': 59.44, 'close': 59.66})
    # uos.feed({'date': '23.Kas.10', 'high': 59.74, 'low': 57.33, 'close': 59.07})
    # uos.feed({'date': '24.Kas.10', 'high': 59.94, 'low': 59.11, 'close': 59.90})
    # uos.feed({'date': '26.Kas.10', 'high': 59.65, 'low': 58.87, 'close': 59.29})
    # uos.feed({'date': '29.Kas.10', 'high': 59.37, 'low': 58.24, 'close': 59.12})
    # uos.feed({'date': '30.Kas.10', 'high': 60.21, 'low': 58.26, 'close': 59.68})
    # uos.feed({'date': '1.Kas.10', 'high': 61.70, 'low': 60.58, 'close': 61.48})

    mfi = MoneyFlowIndex(period=14)

    mfi.feed({'date': '3-Dec-10', 'high': 24.83, 'low': 24.32, 'volume': 18730, 'close': 24.75})
    mfi.feed({'date': '6-Dec-10', 'high': 24.76, 'low': 24.60, 'volume': 12272, 'close': 24.71})
    mfi.feed({'date': '7-Dec-10', 'high': 25.16, 'low': 24.78, 'volume': 24691, 'close': 25.04})
    mfi.feed({'date': '8-Dec-10', 'high': 25.58, 'low': 24.95, 'volume': 18358, 'close': 25.55})
    mfi.feed({'date': '9-Dec-10', 'high': 25.68, 'low': 24.81, 'volume': 22964, 'close': 25.07})
    mfi.feed({'date': '10-Dec-10', 'high': 25.34, 'low': 25.06, 'volume': 15919, 'close': 25.11})
    mfi.feed({'date': '13-Dec-10', 'high': 25.29, 'low': 24.85, 'volume': 16067, 'close': 24.89})
    mfi.feed({'date': '14-Dec-10', 'high': 25.13, 'low': 24.75, 'volume': 16568, 'close': 25.00})
    mfi.feed({'date': '15-Dec-10', 'high': 25.28, 'low': 24.93, 'volume': 16019, 'close': 25.05})
    mfi.feed({'date': '16-Dec-10', 'high': 25.39, 'low': 25.03, 'volume': 9774, 'close': 25.34})
    mfi.feed({'date': '17-Dec-10', 'high': 25.54, 'low': 25.05, 'volume': 22573, 'close': 25.06})
    mfi.feed({'date': '20-Dec-10', 'high': 25.60, 'low': 25.06, 'volume': 12987, 'close': 25.45})
    mfi.feed({'date': '21-Dec-10', 'high': 25.74, 'low': 25.54, 'volume': 10907, 'close': 25.56})
    mfi.feed({'date': '22-Dec-10', 'high': 25.72, 'low': 25.46, 'volume': 5799, 'close': 25.56})
    mfi.feed({'date': '23-Dec-10', 'high': 25.67, 'low': 25.29, 'volume': 7395, 'close': 25.41})
    mfi.feed({'date': '27-Dec-10', 'high': 25.45, 'low': 25.17, 'volume': 5818, 'close': 25.37})
    mfi.feed({'date': '28-Dec-10', 'high': 25.32, 'low': 24.92, 'volume': 7165, 'close': 25.04})
    mfi.feed({'date': '29-Dec-10', 'high': 25.26, 'low': 24.91, 'volume': 5673, 'close': 24.92})
    mfi.feed({'date': '30-Dec-10', 'high': 25.04, 'low': 24.83, 'volume': 5625, 'close': 24.88})
    mfi.feed({'date': '31-Dec-10', 'high': 25.01, 'low': 24.71, 'volume': 5023, 'close': 24.97})
    mfi.feed({'date': '3-Jan-11', 'high': 25.31, 'low': 25.03, 'volume': 7457, 'close': 25.05})
    mfi.feed({'date': '4-Jan-11', 'high': 25.12, 'low': 24.34, 'volume': 11798, 'close': 24.45})
    mfi.feed({'date': '5-Jan-11', 'high': 24.69, 'low': 24.27, 'volume': 12366, 'close': 24.57})
    mfi.feed({'date': '6-Jan-11', 'high': 24.55, 'low': 23.89, 'volume': 13295, 'close': 24.02})
    mfi.feed({'date': '7-Jan-11', 'high': 24.27, 'low': 23.78, 'volume': 9257, 'close': 23.88})
    mfi.feed({'date': '10-Jan-11', 'high': 24.27, 'low': 23.72, 'volume': 9691, 'close': 24.20})
    mfi.feed({'date': '11-Jan-11', 'high': 24.60, 'low': 24.20, 'volume': 8870, 'close': 24.28})
    mfi.feed({'date': '12-Jan-11', 'high': 24.48, 'low': 24.24, 'volume': 7169, 'close': 24.33})
    mfi.feed({'date': '13-Jan-11', 'high': 24.56, 'low': 23.43, 'volume': 11356, 'close': 24.44})
    mfi.feed({'date': '14-Jan-11', 'high': 25.16, 'low': 24.27, 'volume': 13379, 'close': 25.00})


