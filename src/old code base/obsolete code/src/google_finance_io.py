import pandas as pd
import datetime
import os
import numpy as np


def get_one_year_data(which_stock, start, end, verbose=False):
    if end.year - start.year > 1:
        raise Exception('get_one_year_data only supports max 1 year range')

    start_month_abbv = start.strftime('%b')
    start_day = start.day
    start_year = start.year

    end_month_abbv = end.strftime('%b')
    end_day = end.day
    end_year = end.year

    if verbose:
        print("retrieving {} {} data...".format(start_year, which_stock))

    # read raw data from google finance
    df = pd.read_csv('http://finance.google.com/finance/historical?q={stock_name}&'
                     'startdate={start_month_abbv}+{start_day}%2C+{start_year}&'
                     'enddate={end_month_abbv}+{end_day}%2C+{end_year}&output=csv'
                     .format(stock_name=which_stock,
                             start_month_abbv=start_month_abbv,
                             start_day=start_day,
                             start_year=start_year,
                             end_month_abbv=end_month_abbv,
                             end_day=end_day,
                             end_year=end_year
                             ))
    # change date formatting
    df.Date = df.Date.apply(lambda x: datetime.datetime.strptime(x, '%d-%b-%y').date())
    df = df.loc[(df.Date <= end) & (df.Date >= start)]
    if df.empty:
        if verbose:
            print("returning None cuz df is empty".format(start_year))
        return None
    lower_bound_date = start + datetime.timedelta(days=7)
    upper_bound_date = end - datetime.timedelta(days=7)

    last_date = df.Date.iloc[0]
    first_date = df.Date.iloc[-1]

    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    if first_date > lower_bound_date or last_date < upper_bound_date:
        if verbose:
            print("returning None cuz df date boundaries are not satisfied".format(start_year))
        return None
    return df


def data_getter(which_stock, start, end, verbose=False):
    """
    helper function for google finance api
    google finance only gives nearly 16 years data. 
    so if we want more than 16 year, need to concat multiple dataframe.
    my decision is creating 1 year data getter and calling it multiple times.
    :type start datetime.date
    :type end datetime.date
    :type which_stock str
    :returns pandas dataframe
    """

    final_df = None
    # split into chunks if more than 1 years
    cur_start = datetime.date(start.year - 1, start.month, start.day)
    while (True):
        cur_start = datetime.date(cur_start.year + 1, cur_start.month, cur_start.day)
        cur_end = datetime.date(cur_start.year + 1, cur_start.month, cur_start.day)

        if (cur_end.year > end.year):
            cur_end = end

        # get one year data as pandas.df
        one_year_df = get_one_year_data(which_stock, cur_start, cur_end, verbose=verbose)
        # if one year data not proper then go to the next year
        if one_year_df is None:
            continue

        if final_df is None:
            final_df = one_year_df
        else:
            final_df = pd.concat([get_one_year_data(which_stock, cur_start, cur_end), final_df])

        if cur_end == end:
            break

    final_df.reset_index(inplace=True, drop=True)
    return final_df


def missing_value_fix(row):
    if row.open == '-':
        row.open = row.close
        row.high = row.close
        row.low = row.close
        row.volume = 0
    return row


def download_data(stock_names, start_date, end_date, path="../input/raw_data", verbose=False):
    """Download raw data from google finance"""

    if (path != "") and (not os.path.exists(path)):
        os.makedirs(path)

    for stock_name in stock_names:

        if (path != "") and (stock_name + ".csv" in os.listdir(path)):
            if verbose:
                print(stock_name + " already exists.")
            continue

        # Open High Low Close Volume Adj Close
        stock = data_getter(stock_name, start_date, end_date, verbose=verbose)
        if stock is not None:
            # reverse data to get order like 2000-2017
            stock = stock.iloc[::-1]

            # google finance does not have adjusted close. their values already adjusted
            # however we have used yahoo finance and its adjusted close.
            # so i will just copy close value to adjusted close
            stock['adjusted_close'] = stock['close']

            # i found that some data rows are missing with '-'.
            # i will copy adjusted close value to other columns
            stock = stock.apply(missing_value_fix, axis=1)
            stock.open = stock.open.apply(lambda x: np.nan if x == 0 else x)
            stock.open = stock.open.fillna(method='pad')
            stock.volume = stock.volume.apply(lambda x: np.nan if x == 0 else x)
            stock.volume = stock.volume.fillna(method='pad')
            stock = stock.drop_duplicates(subset=["date"])

            # i found that stock.open and others are object dtype.
            # lets be sure that all are float
            stock[['open', 'high', 'low', 'close', 'volume', 'adjusted_close']] = stock[
                ['open', 'high', 'low', 'close', 'volume', 'adjusted_close']].astype('float64')

            # save as a csv file
            if verbose:
                print("creating {}.csv...".format(stock_name))

            if path != "":
                stock.to_csv(path + "/{}.csv".format(stock_name), index=False)
            else:
                return stock


# web.DataReader(which_stock, 'yahoo', start=start, end=end)

if "__name__" == "__main__":
    start_date = datetime.date(1994, 1, 1)
    end_date = datetime.date(2016, 12, 30)
    df = data_getter("SPY", start_date, end_date, verbose=True)
# todo: open values sometimes get 0.0 value. Check and fix later.


# http://www.google.com/finance/historical?q=NYSEARCA:SPY&startdate=Jun+13%2C+2010&enddate=Jun+12%2C+2017&num=30&output=csv

