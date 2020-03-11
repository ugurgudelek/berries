import pandas as pd
class Data:
    def __init__(self):
        # todo: implement attributes later
        pass

    def to_series(self):
        """Deprecated"""
        self.series = pd.Series()
        self.series = self.series.append(self.day_data.iloc[0])
        self.series['label'] = self.raw_label

        # to fix conflict with close
        self.metric_data.rename({'close': 'close_stationary', 'volume': 'volume_metric'}, inplace=True)

        self.series = self.series.append(self.metric_data)
        self.series['label_stationary'] = self.label

        self.series = self.series.append(pd.Series(self.image))

        return self.series


class DataHolder:
    def __init__(self):
        self.storage = pd.DataFrame()

    def append(self, data):
        self.storage = self.storage.append(data, ignore_index=True)

    def save(self, path):
        print("Saving to the {}".format(path))
        self.storage.to_csv(path)

    def reset_storage(self):
        self.storage = pd.DataFrame()


    def to_dataframe(self):
        """Deprecated"""
        print("Constructing dataframe ...")
        self.dataframe = pd.DataFrame()
        for i,data in enumerate(self.storage):
            print(i,len(self.storage))
            if self.dataframe.shape[0] == 0: # it is empty
                self.dataframe = pd.DataFrame(columns=data.to_series().keys().values)
            self.dataframe = self.dataframe.append(data.to_series(), ignore_index=True)
        return self.dataframe