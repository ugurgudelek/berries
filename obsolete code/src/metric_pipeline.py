import numpy as np
import pandas as pd


def ema(data, period):
    """exponentially smoothing moving average"""
    # data needs to be just one column not whole data
    EMA = []

    previous_avg = np.mean(data[0:period])
    EMA.append(previous_avg)
    for datum in data[period:]:
        curr = (previous_avg * (period - 1) + datum) / period
        EMA.append(curr)
        previous_avg = curr

    return EMA


def rsi(data, period=15):
    """relative strength index
    :param data: adjusted_close
    :type data: np.array
    :param period: period_in_days
    :type period: int
    :rtype: np.array
    """
    data = data['adjusted_close'].values
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

    gain = []
    loss = []

    for datum in p_n_list:
        if datum >= 0:
            gain.append(datum)
            loss.append(0)
        else:
            loss.append(-datum)
            gain.append(0)

    # first period's data
    gain_ema = ema(gain, period)
    loss_ema = ema(loss, period)

    RS = np.divide(gain_ema, loss_ema)

    RSI = np.subtract(100, np.divide(100, np.add(1, RS)))
    # RSI = 100 - (100 / (1 + RS))

    return RSI

df = pd.read_csv("../sanity_new/train/stock_with_metrics/spy.csv")

print()