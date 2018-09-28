import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


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


def macd(data, period_long=26, period_short=12):
    """moving average convergence divergence"""

    if period_long <= period_short:
        raise ValueError("period_long should be bigger than period_short")

    ema_long = ema(data['adjusted_close'], period=period_long)
    ema_short = ema(data['adjusted_close'], period=period_short)[(period_long - period_short):]

    return np.subtract(ema_short, ema_long)


# macd trigger custom
# http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
def macd_trigger(data, period_signal=9, period_long=26, period_short=12):

    macd_line = macd(data, period_long, period_short)
    signal_line = ema(macd_line, period_signal)
    macd_histogram = np.subtract(macd_line[period_signal - 1:], signal_line)

    #    plt.plot(macd_line[period_signal-1:100], c='b')
    #    plt.plot(signal_line[0:100], c='r')
    #    plt.bar(left=list(range(100)),height=macd_histogram[0:100], color='g')

    #    plt.show()

    return macd_histogram


def sma(data, period=15):
    """simple moving average"""

    data = data['adjusted_close'].values
    lower = 0
    upper = period

    SMA = []

    for i in data[period - 1:]:
        SMA.append(np.mean(data[lower:upper]))
        lower += 1
        upper += 1

    return np.asarray(SMA)


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


def williamsR(data, period_in_days=14):
    "Calculates the Williams %R indicator."

    result = []

    for curInd in range(period_in_days - 1, data['adjusted_close'].shape[0]):
        # current close
        curClose = data['close'].values[curInd]

        # find the highest high and the lowest low in the period
        highestHigh = np.amax(data['high'].values[curInd - period_in_days + 1: curInd + 1])
        lowestLow = np.amin(data['low'].values[curInd - period_in_days + 1: curInd + 1])

        # calculate %R
        wR = (highestHigh - curClose) / (highestHigh - lowestLow) * (-100)
        result.append(wR)

    return np.asarray(result)


def kdDiff(data, period_in_days=14):
    "Calculates the difference between %K and %D."

    Kpc = []
    Dpc = []

    dpcPeriod = 3

    # calculate %K
    for curInd in range(period_in_days - 1, data['adjusted_close'].shape[0]):
        # current close
        curClose = data['close'].values[curInd]

        # find the highest high and the lowest low in the period
        highestHigh = np.amax(data['high'].values[curInd - period_in_days + 1: curInd + 1])
        lowestLow = np.amin(data['low'].values[curInd - period_in_days + 1: curInd + 1])

        Kpc.append((highestHigh - curClose) / (highestHigh - lowestLow) * 100)

    # calculate %D
    for curInd in range(dpcPeriod - 1, len(Kpc)):
        Dpc.append(np.mean(Kpc[curInd - dpcPeriod + 1: curInd]))

    return np.subtract(Kpc[dpcPeriod - 1: len(Kpc)], Dpc)


def ulOs(data, period1=7, period2=14, period3=28):
    "Calculates the ultimate oscillator. Periods should be from low to high."

    bp = []  # buying pressure
    tr = []  # true range
    uos = []  # ultimate oscillator
    weight1 = period3 / period1
    weight2 = period3 / period2
    weight3 = period3 / period3

    # calculate buying pressure and true range
    for curInd in range(1, data['adjusted_close'].shape[0]):
        curClose = data['close'].values[curInd]
        prClose = data['close'].values[curInd - 1]
        curLow = data['low'].values[curInd]
        curHigh = data['high'].values[curInd]

        bp.append(curClose - np.amin([curLow, prClose]))
        tr.append(np.amax([curHigh, prClose]) - np.amin([curLow, prClose]))

        # calculate the averages and the ultimate oscillator
    for curInd in range(1, data['adjusted_close'].shape[0]):

        avg1value = 0
        avg2value = 0
        avg3value = 0
        uosvalue = 0

        # zeros will be appended if the index is lower than period3

        if curInd >= period1:
            avg1value = np.sum(bp[curInd - period1 + 1: curInd + 1]) / np.sum(tr[curInd - period1 + 1: curInd + 1])

        if curInd >= period2:
            avg2value = np.sum(bp[curInd - period2 + 1: curInd + 1]) / np.sum(tr[curInd - period2 + 1: curInd + 1])

        if curInd >= period3:
            avg3value = np.sum(bp[curInd - period3 + 1: curInd + 1]) / np.sum(tr[curInd - period3 + 1: curInd + 1])
            uosvalue = 100 * ((weight1 * avg1value) + (weight2 * avg2value) + (weight3 * avg3value)) / (
            weight1 + weight2 + weight3)
            uos.append(uosvalue)

    return np.asarray(uos)


def mfi(data, period_in_days=14):
    """Calculates the money flow index for the given period."""


    prmf = [0]  # positive raw money flow
    nrmf = [0]  # negative raw money flow
    mfr = []  # money flow ratio

    # calculate raw money flow
    for curInd in range(1, data['adjusted_close'].shape[0]):

        prTypicalPrice = (data['high'].values[curInd - 1] + data['low'].values[curInd - 1] + data['close'].values[curInd - 1]) / 3
        curTypicalPrice = (data['high'].values[curInd] + data['low'].values[curInd] + data['close'].values[curInd]) / 3

        if curTypicalPrice < prTypicalPrice:
            nrmf.append(curTypicalPrice * data['volume'].values[curInd])
            prmf.append(0)
        else:
            prmf.append(curTypicalPrice * data['volume'].values[curInd])
            nrmf.append(0)

    # calculate money flow ratio
    for curInd in range(period_in_days, data['adjusted_close'].shape[0]):
        sumPosFlow = np.sum(prmf[curInd - period_in_days + 1 : curInd + 1])
        sumNegFlow = np.sum(nrmf[curInd - period_in_days + 1 : curInd + 1])
        if sumNegFlow == 0:
            mfr.append(float("Inf"))
        else:
            mfr.append(sumPosFlow / sumNegFlow)
        

    # calculate and return money flow index
    return np.subtract(100, np.divide(100, np.add(1, mfr)))



