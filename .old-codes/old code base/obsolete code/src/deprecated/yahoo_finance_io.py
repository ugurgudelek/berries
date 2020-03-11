import pandas as pd
import datetime

def data_getter(which_stock, start, end):
    """:type start datetime.date
    :type end datetime.date
    :type which_stock str
    return pandas dataframe"""
    df = pd.read_csv('http://ichart.finance.yahoo.com/table.csv?s={}&g=d'.format(which_stock))
    df = df.loc[(df.Date < str(end)) & (df.Date >= str(start))]
    lower_bound_date = start + datetime.timedelta(days=7)
    upper_bound_date = end - datetime.timedelta(days=7)

    last_date = df.Date.iloc[0]
    first_date = df.Date.iloc[-1]

    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']
    if  first_date > str(lower_bound_date) or last_date < str(upper_bound_date):
        return None
    return df
#  web.DataReader(which_stock, 'yahoo', start=start, end=end)

if "__name__" == "__main__":
    start_date = datetime.date(2000, 1, 1)
    end_date = datetime.date(2002, 12, 30)
    data_getter("SPY", start_date, end_date)
