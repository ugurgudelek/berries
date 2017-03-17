import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

# exponentially smoothing moving average
def ema(data, period):
    EMA = []
    
    previous_avg = np.mean(data[0:period])
    EMA.append(previous_avg)
    for datum in data[period:]:
        curr = (previous_avg * (period - 1) + datum) / period
        EMA.append(curr)
        previous_avg = curr

    # append zeros before the beginning of the period
    for curInd in range(0, period - 1):
        EMA.insert(0,0)

    return np.reshape(np.asarray(EMA), (len(EMA), 1))


# moving average convergence divergence
def macd(data, period_long, period_short):
    if period_long <= period_short:
        raise ValueError("period_long should be bigger than period_short")

    ema_long = ema(data, period=period_long)
    ema_short = ema(data, period=period_short)#[(period_long - period_short):]

    return np.subtract(ema_short, ema_long)


# macd trigger custom
# http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd
def macd_trigger(data, period_signal, period_long, period_short):
    macd_line = macd(data, period_long, period_short)
    signal_line = ema(macd_line, period_signal)
    macd_histogram = np.subtract(macd_line, signal_line)
    #macd_histogram = np.subtract(macd_line[period_signal-1:], signal_line)

#    plt.plot(macd_line[period_signal-1:100], c='b')
#    plt.plot(signal_line[0:100], c='r')
#    plt.bar(left=list(range(100)),height=macd_histogram[0:100], color='g')

#    plt.show()

    return macd_histogram


# simple moving average
def sma(data, period):
    lower = 0
    upper = period

    SMA = []

    for i in data[period - 1:]:
        SMA.append(np.mean(data[lower:upper]))
        lower += 1
        upper += 1

    # append zeros before the beginning of the period
    for curInd in range(0, period - 1):
        SMA.insert(0,0)

    return np.reshape(np.asarray(SMA), (len(SMA), 1))


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
    # get rid of nan values resulted from division by 0
    RS = np.nan_to_num(RS)

    RSI = np.subtract(100, np.divide(100, np.add(1, RS)))
    RSI = np.insert(RSI, 0, 0, axis = 0)
    # RSI = 100 - (100 / (1 + RS))

    return RSI

def williamsR(data, period_in_days = 14):
    "Calculates the Williams %R indicator."

    result = []

    # append zeros before the beginning of the period
    for curInd in range(0, period_in_days - 1):
        result.append(0)
    
    for curInd in range(period_in_days - 1, data.shape[0]):

        # current close
        curClose = data[curInd, 4]
        
        # find the highest high and the lowest low in the period
        highestHigh = np.amax(data[curInd - period_in_days + 1 : curInd + 1, 2])
        lowestLow = np.amin(data[curInd - period_in_days + 1 : curInd + 1, 3])

        # calculate %R
        wR = (highestHigh - curClose) / (highestHigh - lowestLow) * (-100)
        result.append(wR)

    return np.reshape(np.asarray(result), (len(result), 1))

def kdDiff(data, period_in_days = 14):
    "Calculates the difference between %K and %D."

    Kpc = []
    Dpc = []

    dpcPeriod = 3

    # append zeros before the beginning of the period
    for curInd in range(0, period_in_days - 1):
        Kpc.append(0)
        Dpc.append(0)

    # calculate %K
    for curInd in range(period_in_days - 1, data.shape[0]):

        # current close
        curClose = data[curInd, 4]
        
        # find the highest high and the lowest low in the period
        highestHigh = np.amax(data[curInd - period_in_days + 1 : curInd + 1, 2])
        lowestLow = np.amin(data[curInd - period_in_days + 1 : curInd + 1, 3])

        Kpc.append((highestHigh - curClose) / (highestHigh - lowestLow) * 100)

        if curInd >= period_in_days + dpcPeriod - 2 :
            Dpc.append(np.mean(Kpc[curInd - dpcPeriod + 1 : curInd + 1]))
        else:
            Dpc.append(0)
            
    return np.reshape(np.asarray(Kpc), (len(Kpc), 1)) - np.reshape(np.asarray(Dpc), (len(Dpc), 1))


def ulOs(data, period1 = 7, period2 = 14, period3 = 28):
    "Calculates the ultimate oscillator. Periods should be from low to high."

    bp = [0]     # buying pressure
    tr = [0]     # true range
    uos = [0]    # ultimate oscillator
    weight1 = period3 / period1
    weight2 = period3 / period2
    weight3 = period3 / period3
    
    # calculate buying pressure and true range
    for curInd in range(1, data.shape[0]):

        curClose = data[curInd, 4]
        prClose = data[curInd - 1 , 4]
        curLow = data[curInd, 3]
        curHigh = data[curInd, 2]

        bp.append(curClose - np.amin([curLow, prClose]))
        tr.append(np.amax([curHigh, prClose]) - np.amin([curLow, prClose]))        

    # calculate the averages and the ultimate oscillator
    for curInd in range(1, data.shape[0]):

        avg1value = 0
        avg2value = 0
        avg3value = 0
        uosvalue = 0

        # zeros will be appended if the index is lower than period3

        if curInd >= period1:

            avg1value = np.sum(bp[curInd - period1 + 1 : curInd + 1]) / np.sum(tr[curInd - period1 + 1 : curInd + 1])

        if curInd >= period2:

            avg2value = np.sum(bp[curInd - period2 + 1 : curInd + 1]) / np.sum(tr[curInd - period2 + 1 : curInd + 1])

        if curInd >= period3:

            avg3value = np.sum(bp[curInd - period3 + 1 : curInd + 1]) / np.sum(tr[curInd - period3 + 1 : curInd + 1])
            uosvalue = 100 * ((weight1 * avg1value) + (weight2 * avg2value) + (weight3 * avg3value)) / (weight1 + weight2 + weight3)

        uos.append(uosvalue)

    return np.reshape(np.asarray(uos), (len(uos), 1))

def mfi(data, period_in_days = 14):
    "Calculates the money flow index for the given period."

    prmf = [0]    # positive raw money flow
    nrmf = [0]    # negative raw money flow 
    mfr = [0]     # money flow ratio
    
    # calculate raw money flow
    for curInd in range(1, data.shape[0]):

        prTypicalPrice = (data[curInd - 1, 2] + data[curInd - 1, 3] + data[curInd - 1, 4]) / 3
        curTypicalPrice = (data[curInd, 2] + data[curInd, 3] + data[curInd, 4]) / 3

        if curTypicalPrice < prTypicalPrice:
            nrmf.append(curTypicalPrice * data[curInd, 5])
            prmf.append(0)
        else:
            prmf.append(curTypicalPrice * data[curInd, 5])
            nrmf.append(0)

    # append zeros to money flow ratio
    for curInd in range(1, period_in_days):
        mfr.append(0)

    # calculate money flow ratio
    for curInd in range(period_in_days, data.shape[0]):

        sumPosFlow = np.sum(prmf[curInd - period_in_days + 1 : curInd + 1])
        sumNegFlow = np.sum(nrmf[curInd - period_in_days + 1 : curInd + 1])
        mfr.append(sumPosFlow / sumNegFlow)

    # reshape money flow ratio
    mfr = np.reshape(mfr, (len(mfr), 1))

    # calculate and return money flow index
    return 100 - 100 / (1 + mfr)


def main():
    # Date Open High Low Close Volume Adj Close
    spy_data = pd.read_csv("spy.csv").as_matrix()[:, :][::-1]

    # adjust the prices according to adjusted close
    adj_ratio = spy_data[:,6] / spy_data[:,4]
    spy_data[:,1] = adj_ratio * spy_data[:,1]  # open
    spy_data[:,2] = adj_ratio * spy_data[:,2]  # high
    spy_data[:,3] = adj_ratio * spy_data[:,3]  # low
    spy_data[:,4] = spy_data[:,6]              # close is adjusted

    rsi_15_data = rsi(spy_data[:, 6], 15)
    sma_15_data = sma(spy_data[:, 6], 15)
    macd_15_5_data = macd(spy_data[:, 6], 26, 12)
    macd_trigger_9_15_5 = macd_trigger(spy_data[:, 6], 9, 26, 12)
    willR = williamsR(spy_data)
    kdHist = kdDiff(spy_data)
    ultimateOs = ulOs(spy_data)
    mfIndex = mfi(spy_data)

    result = np.concatenate((rsi_15_data, sma_15_data, macd_15_5_data, macd_trigger_9_15_5, willR, kdHist, ultimateOs, mfIndex), axis = 1)

    np.savetxt('foo.csv', result, delimiter=',')

    # plt.plot(sma_15_data, color='r')
    # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # plt.show()

if __name__ == "__main__":
    main()
