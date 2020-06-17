# -*- coding: utf-8 -*-
# @Time   : 6/16/2020 4:11 PM
# @Author : Ugur Gudelek
# @Email  : ugurgudelek@gmail.com
# @File   : flights.py

from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from berries.utils.transform import Normalizer

from berries.datasets.dataset import StatelessTimeseriesDataset


class Flights:

    def __init__(self):
        self.raw_flight_data = sns.load_dataset('flights')['passengers'].values.astype(float)

        self.normalize = Normalizer().fit(data=self.raw_flight_data)
        self.flight_data = self.normalize.transform(data=self.raw_flight_data)

        self.trainsize = 108

        self.trainset = StatelessTimeseriesDataset(timeseries=self.flight_data[:self.trainsize],
                                                   seq_len=4,
                                                   look_ahead=1)

        self.testset = StatelessTimeseriesDataset(timeseries=self.flight_data[self.trainsize:],
                                                   seq_len=4,
                                                   look_ahead=1)





if __name__ == "__main__":
    import numpy as np

    f = Flights()
    loader = DataLoader(f.testset, batch_size=1, shuffle=False)

    for i, (data, targets) in enumerate(loader):
        print(f"""
        index: {i}
        data: {data}
        targets: {targets}
        """)

