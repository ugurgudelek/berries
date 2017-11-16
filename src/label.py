"""This files containes several function about label retrieval"""
import datetime

class LabelEngine:
    def __init__(self,financeIO, make_stationary=True):
        self.labels = []
        self.financeIO = financeIO

        self.make_stationary = make_stationary

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

            return (current_close - old_close) / current_close

        return current_close
