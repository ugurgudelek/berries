import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt


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
def macd_trigger(data, period_signal, period_long, period_short):
    macd_line = macd(data, period_long, period_short)
    signal_line = ema(macd_line, period_signal)
    macd_histogram = np.subtract(macd_line[period_signal-1:], signal_line)

    plt.plot(macd_line[period_signal-1:100], c='b')
    plt.plot(signal_line[0:100], c='r')
    plt.bar(left=list(range(100)),height=macd_histogram[0:100], color='g')

    plt.show()



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

    rsi_15_data = rsi(data['Adj Close'].values, 15)
    sma_15_data = sma(data['Adj Close'].values, 15)

    macd_15_5_data = macd(data['Adj Close'].values, 26, 12)

    macd_trigger_9_15_5 = macd_trigger(data['Adj Close'].values, 9, 26, 12)

    # plt.plot(sma_15_data, color='r')
    # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # plt.show()


if __name__ == "__main__":
    main()
