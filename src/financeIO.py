"""This file is created for downloading useful data from several finance api"""

import pandas as pd
import datetime
import os


class IO:
    def __init__(self):
        pass

    def get_one_day_data(self, stock_name, date):
        """

        :param str stock_name: stock name
        :param datetime.datetime date: date to data fetch
        :return: 1 day data if data is available else None
        """
        data = self.query(stock_name=stock_name, start_date=date, end_date=date)

        if data.shape[0] == 0:
            return None

        data['name'] = stock_name

        return data[['name', 'date', 'open', 'high', 'low', 'close', 'volume']]

    def get_next_day_data(self, stock_name, date):
        one_day_data = None

        while one_day_data is None:  # loop until finding propor day
            date += datetime.timedelta(days=1)  # next day
            one_day_data = self.get_one_day_data(stock_name=stock_name, date=date)

        return one_day_data


class GoogleFinanceIO(IO):
    def __init__(self):
        IO.__init__(self)

    def query(self, stock_name, start_date, end_date):
        """Query function is the parameterized version of api itself.
        :param str stock_name: stock name
        :param datetime.datetime start_date: start of closed interval
        :param datetime.datetime end_date: end of closed interval
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

        # todo: maybe implement data fix
        data = pd.read_csv(q).iloc[::-1]  # reverse
        data = data.rename(columns=lambda x:x.lower())
        data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(x, '%d-%b-%y'))
        return data

    def download_data(self, stock_names, start_date, end_date, path="../input/raw_input", redownload=False, verbose=False):
        """Download raw data from google finance"""

        if not os.path.exists(path):  # check if path exists
            os.makedirs(path)

        save_start_date = start_date
        for stock_name in stock_names:  # for each stock
            if not redownload:  # do not download if file exists
                if os.path.exists(os.path.join(path, stock_name+".csv")):
                    if verbose:
                        print("Skipping {}...".format(stock_name))
                    continue
            if verbose:
                print("Downloading {}...".format(stock_name))
            start_date = save_start_date
            stock_data = pd.DataFrame()
            # download one year at a time because google io does not allow to download all at once.
            while start_date < end_date:
                next_year_start_date = start_date.replace(year=start_date.year + 1)
                if next_year_start_date > end_date:
                    one_year_data = self.query(stock_name=stock_name, start_date=start_date,
                                               end_date=end_date)
                else:
                    one_year_data = self.query(stock_name=stock_name, start_date=start_date,
                                               end_date=next_year_start_date)

                stock_data = pd.concat((stock_data, one_year_data))  # merge data

                # day + 1 to not download same day again
                start_date = next_year_start_date.replace(day=next_year_start_date.day + 1)

            stock_data.to_csv(os.path.join(path,stock_name + ".csv"), index=False)


class YahooFinanceIO(IO):
    # todo: Implementation
    def __init__(self):
        IO.__init__(self)
        pass

    def query(self, stock_name, start_date, end_date):
        pass


class LocalIO(IO):
    def __init__(self, filepath="../input/raw_input"):
        IO.__init__(self)
        self.filepath = filepath

        def remove_extension(filename):
            return filename.split('.')[0]

        def read_all_files(filepath):
            dic = dict()
            filenames = os.listdir(filepath)
            for filename in filenames:
                full_filename = os.path.join(self.filepath, filename)
                df = pd.read_csv(full_filename, index_col=None)
                df['date'] = df['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
                dic[remove_extension(filename)] = df
            return dic

        self.localdata = read_all_files(self.filepath)

    def query(self, stock_name, start_date, end_date):
        """Query function is the parameterized version of api itself.
        :param str stock_name: stock name
        :param datetime.datetime start_date: start of closed interval
        :param datetime.datetime end_date: end of closed interval
        :return pandas.DataFrame closed interval of stock_name historical data
        with the boundaries defined start_data and end_date
        and signature is:
        Date,Open,High,Low,Close,Volume
        """

        stockdata = pd.DataFrame(self.localdata[stock_name])

        r = stockdata.loc[((stockdata['date'] >= start_date) & (stockdata['date'] <= end_date))]

        return r


if __name__ == "__main__":
    # date = datetime.datetime.strptime('03-10-2016', '%d-%m-%Y')
    # google = GoogleFinanceIO()
    # print(google.get_one_day_data('spy', date))
    # print(google.get_next_day_data('spy', date))
    #
    # local = LocalIO(filepath="../input/raw_input")
    # print(local.get_one_day_data('spy', date))
    # print(local.get_next_day_data('spy', date))

    DATE_FORMAT = '%d-%m-%Y'
    google = GoogleFinanceIO()
    stock_names = ['spy', 'xlf', 'xlu', 'xle',
                   'xlp', 'xli', 'xlv', 'xlk', 'ewj',
                   'xlb', 'xly', 'eww', 'dia', 'ewg',
                   'ewh', 'ewc', 'ewa']
    google.download_data(stock_names=stock_names,
                         start_date=datetime.datetime.strptime('01-01-2000', DATE_FORMAT).date(),
                         end_date=datetime.datetime.strptime('31-12-2016', DATE_FORMAT).date(), verbose=True)

