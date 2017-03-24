import pandas_datareader as web
import datetime

def data_getter(which_stock, start, end):
    return web.DataReader(which_stock, 'yahoo', start=start, end=end)


if "__name__" == "__main__":
    start_date = datetime.datetime(2000, 1, 1)
    end_date = datetime.datetime(2002, 12, 30)
    data_getter("SPY", start_date, end_date)
