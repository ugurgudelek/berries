import pandas as pd
import os
import numpy as np
import preprocessing


class TestModule:
    def __init__(self, filepath, stock_names):
        # read all test data
        self.data = pd.DataFrame()
        for stock_name in stock_names:
            reading = pd.read_csv(os.path.join(filepath, stock_name + '.csv'))
            reading['name'] = reading.shape[0] * [stock_name]
            reading = preprocessing.apply_normalization_to_raw_data(reading)
            self.data = pd.concat((self.data, reading))
        # sort according to date.
        self.data = self.data.sort_values(['date','name'])
        # reset index
        self.data = self.data.reset_index(drop=True)

        # store  days
        def date_gen(dates):
            for date in dates:
                yield date
        self._dates = date_gen(self.data['date'].unique())

        # store now current day
        self.current_day = next(self._dates)


    def get_next_day(self):
        """First it updates current_day then returns data_packet of updated day"""
        self.current_day = next(self._dates)
        return self.data[self.data['date'] == self.current_day]

    def get_current_day(self):
        """:return data packet of current_day"""
        return self.data[self.data['date'] == self.current_day]

    def evalute(self, predictions, date):
        """:return evaluation result"""
        # todo: check pct_change_tanh
        actuals = self.data[self.data['date'] == date]['pct_change_tanh']
        res = [1 if a*p > 0 else 0 for (a,p) in zip(actuals, predictions)]
        return sum(res) / len(res)





# test
stock_names = ['spy', 'xlf', 'xlu', 'xle',
               'xlp', 'xli', 'xlv', 'xlk', 'ewj',
               'xlb', 'xly', 'eww', 'dia', 'ewg',
               'ewh', 'ewc', 'ewa']
data = TestModule("../input/raw_data", stock_names)
c_day = data.get_current_day()
n_day = data.get_next_day()
rand = (np.random.rand(17,1).reshape(17,) * 2 ) - 1
result = data.evalute(rand, '2000-01-04')
print(result)