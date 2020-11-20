import numpy as np


def open_data(direc, ratio_train=0.8, dataset="ECG5000"):
    """Input:
    direc: location of the UCR archive
    ratio_train: ratio to split training and testset
    dataset: name of the dataset in the UCR archive"""
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.loadtxt(datadir + '_TRAIN', delimiter=',')
    data_test_val = np.loadtxt(datadir + '_TEST', delimiter=',')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)
    data = np.expand_dims(data, -1)

    N, D, _ = data.shape

    ind_cut = int(ratio_train * N)
    ind = np.random.permutation(N)
    # yapf: disable
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]
    # yapf: enable


def slide_data(data, window, stride=1):
    # data: pd.DataFrame
    # data.index: time
    # f1,f2,f2 .... label are columns

    X_list = list()
    y_list = list()
    date_list = list()

    data_dates = list(data.index)

    for start_ix in range(0, data.shape[0] - window + 1, stride):
        end_ix = start_ix + window
        X_list.append(data.iloc[start_ix:end_ix, :-1].values)
        y_list.append(data.iloc[start_ix:end_ix, -1].values)
        date_list.append(data_dates[start_ix:end_ix])

    X = np.array(X_list)
    y = np.array(y_list)
    dates = np.array(date_list)
    return X, y[:, -1].reshape(-1, 1), dates[:, -1].reshape(-1, 1)


def split_data(X, y, dates, train_ratio):
    N, D, _ = X.shape

    ind_cut = int(train_ratio * N)
    # ind = np.random.permutation(N)
    ind = np.array(range(N))
    #yapf: disable
    return (X[ind[:ind_cut], :, :],  X[ind[ind_cut:], :, :],
            y[ind[:ind_cut], :],     y[ind[ind_cut:], :],
            dates[ind[:ind_cut], :], dates[ind[ind_cut:], :])
    #yapf:enable


def label_discreatize(y, bounds=(-0.02, 0.02)):

    classes = [0, 1, 2]

    def apply(label):
        if label <= bounds[0]:
            return classes[0]
        elif label > bounds[1]:
            return classes[2]
        return classes[1]

    y_d = np.array(list(map(apply, y)))

    # print(
    #     f'Class distributions: {((y_d == 0).sum(), (y_d == 1).sum(), (y_d == 2).sum())}'
    # )

    return y_d


# Check if label is 0-based
# base = np.min(y_train)
# if base != 0:
#     y_train -= base
# y_val -= base

if __name__ == "__main__":
    import pandas as pd
    arr = np.arange(0, 200, 1).reshape((20, 10))
    df = pd.DataFrame(arr)
    # print(df)

    slided = slide_data(df, 5, stride=5)

    print(slided)
