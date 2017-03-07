# * Fon: SPY
#
# * Dataset yapısı:
#     - Verinin kendisi		15-30-50-100-200
#     - 5 x RSI 			15-30-50-100-200
#     - 5 x SMA			15-30-50-100-200
#     - 5 x W%R			15-30-50-100-200
#     - 5 x KD			15-30-50-100-200
#     - 5 x MACD			15-30-50-100-200
#     - Interest Rate
#     - Inflation Data
#     - Google Search
#     - Google Search News
#     - Ülke bonosu

def rsi(data, period):
    if period == 1:
        return data

    p_n_list = []
    yesterday_price = data[period-2]
    # drop first periodth data
    for datum in data[period-1:]:
        today_price = datum
        p_n_list.append(today_price - yesterday_price)
        yesterday_price = today_price

    p_sum = 0.0
    n_sum = 0.0

    lower = 0
    upper = lower + period


    p_n_list[lower:upper]

    # TODO : implement rsi


    # # calculate for first frame
    # for i in range(frame_length):
    #     if p_n_list[i] > 0:
    #         p_sum += p_n_list[i]
    #     else:
    #         n_sum += abs(p_n_list[i])
    # rs = [p_sum / n_sum]
    #
    # for i in range(frame_length, len(p_n_list), 1):
    #     # append new element
    #     if p_n_list[i] > 0:
    #         p_sum += p_n_list[i]
    #     else:
    #         n_sum += abs(p_n_list[i])
    #
    #     # remove first element
    #     if p_n_list[i - frame_length] > 0:
    #         p_sum -= p_n_list[i - frame_length]
    #     else:
    #         n_sum -= abs(p_n_list[i - frame_length])
    #
    #     rs.append((p_sum + 1) / (n_sum + 1))
    #
    # rsi = []
    # for _rs in rs:
    #     rsi.append(100.0 - 100.0 / (1 + _rs))
    #
    # return rsi


import csv
import pandas as pd


def load_spy_data():
    _dataset = []
    with open("spy.csv", "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            _dataset.append(line)
    return _dataset[1:]


# Date Open High Low Close Volume Adj Close
dataset = load_spy_data()[::-1]  # reverse dataset order

dframe = pd.DataFrame(dataset, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
data = dframe[['Date', 'Adj Close']]

data['Adj Close'] = pd.DataFrame((data['Adj Close'].values).astype(float))
print(data.head())


rsi_15_data = rsi(data['Adj Close'].values,15)

