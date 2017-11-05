"""This file is created for downloading useful data from several finance api"""

import pandas as pd
from datetime import datetime


class GoogleFinanceIO:
    def __init__(self):
        pass

    def query(self, stock_name, start_date, end_date):
        """Query function is the parameterized version of api itself.
        :param str stock_name: stock name
        :param datetime start_date: start of closed interval
        :param datetime end_date: end of closed interval
        :return pandas.DataFrame closed interval of stock_name historical data 
        with the boundaries defined start_data and end_date
        and signature is:
        Date,Open,High,Low,Close,Volume
        """

        q = 'http://finance.google.com/finance/historical?' \
            'q={stock_name}&' \
            'startdate={start_month_abbv}+{start_day}%2C+{start_year}&' \
            'enddate={end_month_abbv}+{end_day}%2C+{end_year}&' \
            'output=csv'.format(stock_name=stock_name,
                                start_month_abbv=start_date.strftime('%b'),
                                start_day=start_date.day,
                                start_year=start_date.year,
                                end_month_abbv=end_date.strftime('%b'),
                                end_day=end_date.day,
                                end_year=end_date.year
                                )

        return pd.read_csv(q)


    def get_one_day_data(self, stock_name, date):
        """
    
        :param str stock_name: stock name
        :param datetime date: date to data fetch
        :return: 1 day data if data is available else None
        """
        data = self.query(stock_name=stock_name, start_date=date, end_date=date)

        if data.shape[0] == 0:
            return None

        data['Name'] = stock_name

        return data[['Name', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def download_data(self, stock_name,  start_date, end_date):
        # todo: Implementation
        pass



class YahooFinanceIO:
    # todo: Implementation
    pass

class LocalIO:
    # todo: Implementation
    pass



if __name__ == "__main__":
    date = datetime.strptime('03-10-2016', '%d-%m-%Y')
    google = GoogleFinanceIO()
    print(google.get_one_day_data('spy', date))
