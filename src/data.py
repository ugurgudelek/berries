import pandas as pd
class Data:
    def __init__(self):
        self.date = None
        self.stock_name = None
        self.day_data = None
        self.metric_data = None
        self.label = None
        self.feature_data = None
        self.image = None

        self.series = pd.Series()

    def to_series(self):
        self.series['date'] = self.date
        self.series['stock_name'] = self.stock_name
        self.series['open'] = self.day_data['open'].values[0]
        self.series['high'] = self.day_data['high'].values[0]
        self.series['low'] = self.day_data['low'].values[0]
        self.series['close'] = self.day_data['close'].values[0]
        self.series['volume'] = self.day_data['volume'].values[0]
        for (index, values) in self.metric_data.iteritems():
            if index == 'close':
                self.series['close_stationary'] = values
                self.series[index] = values
        self.series['label_stationary'] = self.label

        for i,value in enumerate(self.image):
            self.series['px'+str(i)] = value

        return self.series


class DataHolder:
    def __init__(self):
        self.storage = []

        self.dataframe = pd.DataFrame()

    def append(self, data):
        self.storage.append(data)

    def to_dataframe(self):

        for data in self.storage:
            self.dataframe = self.dataframe.append(data.to_series(), ignore_index=True)
        return self.dataframe