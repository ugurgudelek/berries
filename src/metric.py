"""This file created with the intention of implementation of new rsi,macd,... metrics.
Difference from regular one is these metrics store previous states and we can feed them one by one."""

import pandas as pd

class Metric:
    pass

class RSI:
    """Relative Strength Index (RSI)
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:relative_strength_index_rsi"""

    def __init__(self, period):
        self.period = period

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
        """row should be a series and should have 'date' and 'data' columns
        data = pd.Series('date','data')
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
    pass


class MACD:
    pass


class MACD_Trigger:
    pass


class SMA:
    pass


class WilliamR:
    pass


class KDDiff:
    pass


class UltimateOscillator:
    pass


class MoneyFlowIndex:
    pass


class Bucket:
    """This is variation of queue data structure.
    It holds first few items. Whenever new item added, oldest will be deleted."""

    def __init__(self, size):
        self.size = size
        self.container = []

    def __len__(self):
        return len(self.container)

    def full(self):
        return self.__len__() == self.size

    def put(self, data):
        if self.full():  # throw older item if full
            self.container.pop(0)

        # append new as last item
        self.container.append(data)

    def average(self):
        return sum(self.container) / self.size

    def __str__(self):
        string = ""
        string += str(self.container)
        return string

    def __repr__(self):
        return self.__str__()
