"""This files containes several function about label retrieval"""
import datetime

class Label:
    def __init__(self):
        self.labels = []

    def last_label(self):
        if self.__len__() == 0:
            return None
        return self.labels[-1]

    def __len__(self):
        return len(self.labels)



    def get_label_for(self, stock_name, date, finance_io, take_difference=False, current_close=None):
        one_day_data = None

        while one_day_data is None:  # loop until finding propor day
            date += datetime.timedelta(days=1)  # next day
            one_day_data = finance_io.get_one_day_data(stock_name=stock_name, date=date)

        if take_difference:
            return one_day_data['Close'].values[0] - current_close

        return one_day_data['Close'].values[0]
