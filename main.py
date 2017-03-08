# * Fon: SPY
#
# * Dataset yapısı:
#     - Verinin kendisi		15-30-50-100-200
#     - DONE : 5 x RSI 			15-30-50-100-200
#     - DONE : 5 x SMA			15-30-50-100-200
#     - DONE : 5 x W%R			15-30-50-100-200
#     - DONE : 5 x KD			15-30-50-100-200
#     - DONE : 5 x MACD	(moving average convergence divergence)		15-30-50-100-200
#     - Interest Rate
#     - Inflation Data
#     - Google Search
#     - Google Search News
#     - Ülke bonosu

import numpy as np
import csv
import pandas as pd


# exponentially smoothing moving average
def ema(data, period):
    EMA = []
    previous_avg = np.average(data[0:period])
    EMA.append(previous_avg)
    for datum in data[period:]:
        curr = (previous_avg * (period - 1) + datum) / period
        EMA.append(curr)
        previous_avg = curr
    return EMA


# moving average convergence divergence
def macd(data, period_long, period_short):
    if period_long <= period_short:
        raise ValueError("period_long should be bigger than period_short")

    ema_long = ema(data, period=period_long)
    ema_short = ema(data, period=period_short)[(period_long - period_short):]

    return np.subtract(ema_short, ema_long)

# macd trigger custom
# http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
def macd_trigger(data,period_signal, period_long, period_short):
    macd_line = macd(data,period_long,period_short)
    signal_line = ema(data, period_signal)
    macd_histogram = np.subtract(macd_line - signal_line)

    return macd_histogram


# simple moving average
def sma(data, period):
    lower = 0
    upper = period

    SMA = []

    for i in data[period - 1:]:
        SMA.append(np.average(data[lower:upper]))
        lower += 1
        upper += 1

    return SMA


# relative strength index
def rsi(data, period):
    # period kadar data kaybedeceğiz cünkü;
    # yesterday_price hesaplanırken 1 tane gidiyor.
    # rsi hesaplanırken de period - 1 tane gidiyor.

    p_n_list = []
    yesterday_price = data[0]
    # drop first periodth data
    for datum in data[1:]:
        today_price = datum
        p_n_list.append(today_price - yesterday_price)
        yesterday_price = today_price

    RSs = []

    U = 0.0
    D = 0.0

    # first calculation
    for datum in p_n_list[0:period]:
        if datum >= 0:
            U += datum / period
        else:
            D -= datum / period
    RSs.append(U / D)

    cnt = 1
    for datum in p_n_list[period:]:
        RSs.append((RSs[cnt - 1] * (period - 1) + datum) / period)
        cnt += 1

    RSI = []
    # calculate RSI
    for rs in RSs:
        RSI.append(100 - (100 / (1 + rs)))

    return RSIDONE


def load_spy_data():
    _dataset = []
    with open("spy.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            _dataset.append(line)
    return _dataset[1:]


def main():
    # Date Open High Low Close Volume Adj Close
    dataset = load_spy_data()[::-1]  # reverse dataset order

    dframe = pd.DataFrame(dataset, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
    data = dframe[['Date', 'Adj Close']]

    data['Adj Close'] = pd.DataFrame((data['Adj Close'].values).astype(float))

    print(data.head())

    rsi_15_data = rsi(data['Adj Close'].values, 15)
    print(rsi_15_data)
    sma_15_data = sma(data['Adj Close'].values, 15)

    macd(data['Adj Close'].values, 15, 5)

if __name__ == "__main__":
    main()
