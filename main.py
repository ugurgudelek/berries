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
def macd_trigger(data, period_signal, period_long, period_short):
    macd_line = macd(data, period_long, period_short)
    signal_line = ema(macd_line, period_signal)
    macd_histogram = np.subtract(macd_line[period_signal-1:], signal_line)

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

    return np.asarray(SMA)


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

    RSI = np.subtract(100, np.divide(100, np.add(1, RS)))
    # RSI = 100 - (100 / (1 + RS))

    return RSI

def williamsR(data, period_in_days = 14):
    "Calculates the Williams %R indicator."

    result = []
    
    for curInd in range(period_in_days - 1, data.shape[0]):

        # current close
        curClose = data[curInd, 4]
        
        # find the highest high and the lowest low in the period
        highestHigh = np.amax(data[curInd - period_in_days + 1 : curInd + 1, 2])
        lowestLow = np.amin(data[curInd - period_in_days + 1 : curInd + 1, 3])

        # calculate %R
        wR = (highestHigh - curClose) / (highestHigh - lowestLow) * (-100)
        result.append(wR)

    return np.asarray(result)

def kdDiff(data, period_in_days = 14):
    "Calculates the difference between %K and %D."

    Kpc = []
    Dpc = []

    dpcPeriod = 3

    # calculate %K
    for curInd in range(period_in_days - 1, data.shape[0]):

        # current close
        curClose = data[curInd, 4]
        
        # find the highest high and the lowest low in the period
        highestHigh = np.amax(data[curInd - period_in_days + 1 : curInd + 1, 2])
        lowestLow = np.amin(data[curInd - period_in_days + 1 : curInd + 1, 3])

        Kpc.append((highestHigh - curClose) / (highestHigh - lowestLow) * 100)


    # calculate %D
    for curInd in range(dpcPeriod - 1, len(Kpc)):
        Dpc.append(np.mean(Kpc[curInd - dpcPeriod + 1 : curInd]))
            
    return np.subtract(Kpc[dpcPeriod - 1 : len(Kpc)], Dpc)


def ulOs(data, period1 = 7, period2 = 14, period3 = 28):
    "Calculates the ultimate oscillator. Periods should be from low to high."

    bp = []     # buying pressure
    tr = []     # true range
    uos = []    # ultimate oscillator
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

    return np.asarray(uos)

def mfi(data, period_in_days = 14):
    "Calculates the money flow index for the given period."

    prmf = [0]    # positive raw money flow
    nrmf = [0]    # negative raw money flow 
    mfr = []     # money flow ratio
    
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

    # calculate money flow ratio
    for curInd in range(period_in_days, data.shape[0]):

        sumPosFlow = np.sum(prmf[curInd - period_in_days + 1 : curInd + 1])
        sumNegFlow = np.sum(nrmf[curInd - period_in_days + 1 : curInd + 1])
        mfr.append(sumPosFlow / sumNegFlow)

    # calculate and return money flow index
    return np.subtract(100, np.divide(100, np.add(1, mfr)))


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

    # append zeros at the beginning of the results to make them of uniform size
    total_length = spy_data.shape[0]

    rsi_15_data = np.insert(rsi_15_data, 0, np.zeros(total_length - rsi_15_data.shape[0]))
    sma_15_data = np.insert(sma_15_data, 0, np.zeros(total_length - sma_15_data.shape[0]))
    macd_15_5_data = np.insert(macd_15_5_data, 0, np.zeros(total_length - macd_15_5_data.shape[0]))
    macd_trigger_9_15_5 = np.insert(macd_trigger_9_15_5, 0, np.zeros(total_length - macd_trigger_9_15_5.shape[0]))
    willR = np.insert(willR, 0, np.zeros(total_length - willR.shape[0]))
    kdHist = np.insert(kdHist, 0, np.zeros(total_length - kdHist.shape[0]))
    ultimateOs = np.insert(ultimateOs, 0, np.zeros(total_length - ultimateOs.shape[0]))
    mfIndex = np.insert(mfIndex, 0, np.zeros(total_length - mfIndex.shape[0]))

    # reshape results so that they can be concatenated and written to a file
    rsi_15_data = np.reshape(rsi_15_data, (rsi_15_data.shape[0], 1))
    sma_15_data = np.reshape(sma_15_data, (sma_15_data.shape[0], 1))
    macd_15_5_data = np.reshape(macd_15_5_data, (macd_15_5_data.shape[0], 1))
    macd_trigger_9_15_5 = np.reshape(macd_trigger_9_15_5, (macd_trigger_9_15_5.shape[0], 1))
    willR = np.reshape(willR, (willR.shape[0], 1))
    kdHist = np.reshape(kdHist, (kdHist.shape[0], 1))
    ultimateOs = np.reshape(ultimateOs, (ultimateOs.shape[0], 1))
    mfIndex = np.reshape(mfIndex, (mfIndex.shape[0], 1))

    result = np.concatenate((rsi_15_data, sma_15_data, macd_15_5_data, macd_trigger_9_15_5, willR, kdHist, ultimateOs, mfIndex), axis = 1)

    np.savetxt('foo.csv', result, delimiter=',')

    # plt.plot(sma_15_data, color='r')
    # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # plt.show()

if __name__ == "__main__":
    main()
