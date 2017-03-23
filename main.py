import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import metrics as mt

def crop_data(arr):
    """crop the non-adjusted data starting from the beginning"""
    min_len = len(arr[0])
    for data in arr[1:]:
        min_len = min(min_len, len(data))

    for i in range(len(arr)):
        # crop beginning of data
        arr[i] = arr[i][len(arr[i]) - min_len :]

    return arr


def main():
    # Date Open High Low Close Volume Adj Close
    spy_data = pd.read_csv("spy.csv").as_matrix()[:, :][::-1]

    # adjust the prices according to adjusted close
    adj_ratio = spy_data[:, 6] / spy_data[:, 4]
    spy_data[:, 1] = adj_ratio * spy_data[:, 1]  # open
    spy_data[:, 2] = adj_ratio * spy_data[:, 2]  # high
    spy_data[:, 3] = adj_ratio * spy_data[:, 3]  # low
    spy_data[:, 4] = spy_data[:, 6]  # close is adjusted

    # lets get some data over dataset
    rsi_15_data = mt.rsi(spy_data[:, 6], 15)
    sma_15_data = mt.sma(spy_data[:, 6], 15)
    macd_15_5_data = mt.macd(spy_data[:, 6], 26, 12)
    macd_trigger_9_15_5 = mt.macd_trigger(spy_data[:, 6], 9, 26, 12)
    willR = mt.williamsR(spy_data)
    kdHist = mt.kdDiff(spy_data)
    ultimateOs = mt.ulOs(spy_data)
    mfIndex = mt.mfi(spy_data)

    # created data arr to hold all info
    data_arr = [rsi_15_data,sma_15_data,macd_15_5_data,macd_trigger_9_15_5,willR,kdHist,ultimateOs,mfIndex]

    data_arr = crop_data(arr= data_arr)


    # # append zeros at the beginning of the results to make them of uniform size
    # total_length = spy_data.shape[0]
    #
    # rsi_15_data = np.insert(rsi_15_data, 0, np.zeros(total_length - rsi_15_data.shape[0]))
    # sma_15_data = np.insert(sma_15_data, 0, np.zeros(total_length - sma_15_data.shape[0]))
    # macd_15_5_data = np.insert(macd_15_5_data, 0, np.zeros(total_length - macd_15_5_data.shape[0]))
    # macd_trigger_9_15_5 = np.insert(macd_trigger_9_15_5, 0, np.zeros(total_length - macd_trigger_9_15_5.shape[0]))
    # willR = np.insert(willR, 0, np.zeros(total_length - willR.shape[0]))
    # kdHist = np.insert(kdHist, 0, np.zeros(total_length - kdHist.shape[0]))
    # ultimateOs = np.insert(ultimateOs, 0, np.zeros(total_length - ultimateOs.shape[0]))
    # mfIndex = np.insert(mfIndex, 0, np.zeros(total_length - mfIndex.shape[0]))

    # # reshape results so that they can be concatenated and written to a file
    # rsi_15_data = np.reshape(rsi_15_data, (rsi_15_data.shape[0], 1))
    # sma_15_data = np.reshape(sma_15_data, (sma_15_data.shape[0], 1))
    # macd_15_5_data = np.reshape(macd_15_5_data, (macd_15_5_data.shape[0], 1))
    # macd_trigger_9_15_5 = np.reshape(macd_trigger_9_15_5, (macd_trigger_9_15_5.shape[0], 1))
    # willR = np.reshape(willR, (willR.shape[0], 1))
    # kdHist = np.reshape(kdHist, (kdHist.shape[0], 1))
    # ultimateOs = np.reshape(ultimateOs, (ultimateOs.shape[0], 1))
    # mfIndex = np.reshape(mfIndex, (mfIndex.shape[0], 1))
    #
    # result = np.concatenate(
    #     (rsi_15_data, sma_15_data, macd_15_5_data, macd_trigger_9_15_5, willR, kdHist, ultimateOs, mfIndex), axis=1)
    #
    # np.savetxt('foo.csv', result, delimiter=',')

    # plt.plot(sma_15_data, color='r')
    # plt.bar(left=list(range(len(macd_trigger_9_15_5))),height=-200 - macd_trigger_9_15_5, color='b')
    # plt.scatter(x=list(range(len(macd_trigger_9_15_5))), y=macd_trigger_9_15_5, color='r', s=1)
    # plt.show()


if __name__ == "__main__":
    main()