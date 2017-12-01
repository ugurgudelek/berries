"""This files containes several function about label retrieval"""
import datetime
import numpy as np
import pickle

class LabelEngine:
    def __init__(self,financeIO, make_stationary=True, apply_tanh=True):
        self.labels = []
        self.financeIO = financeIO

        self.make_stationary = make_stationary
        self.apply_tanh = apply_tanh

    def save_instance(self, filepath, run_number):
        filename = filepath+'/{}_label_engine.pkl'.format(run_number)
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def last_label(self):
        if self.__len__() == 0:
            return None
        return self.labels[-1]

    def __len__(self):
        return len(self.labels)



    def get_label_for(self, stock_name, date, old_close=None):

        one_day_data = self.financeIO.get_next_day_data(stock_name, date)
        current_close = one_day_data['close'].values[0]

        if self.make_stationary:
            if old_close is None:
                raise Exception("if make_stationary is True then old_close should be passes.")

            stationary_close = (current_close - old_close) / current_close

            if self.apply_tanh:
                return np.tanh(stationary_close)
            return stationary_close

        return current_close

class Label:
    """Store label related attributes and stationary methods."""
    pass
