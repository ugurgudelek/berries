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
    return data[ind[:ind_cut], 1:, :], data[ind[ind_cut:], 1:, :], data[ind[:ind_cut], 0, :], data[ind[ind_cut:], 0, :]


def slide_data(data, window):

    X_list = list()
    y_list = list()
    for start_ix, (date, row) in enumerate(data.iterrows()):
        end_ix = start_ix + window
        if end_ix > data.shape[0]:
            break
        X_list.append(data.iloc[start_ix: end_ix, :-1].values)
        y_list.append(data.iloc[start_ix: end_ix, -1].values)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y[:, -1][:, np.newaxis]


def split_data(X, y, train_ratio):
    N, D, _ = X.shape

    ind_cut = int(train_ratio * N)
    # ind = np.random.permutation(N)
    ind = np.array(range(N))
    return X[ind[:ind_cut], :, :], X[ind[ind_cut:], :, :], y[ind[:ind_cut], :], y[ind[ind_cut:], :]


def label_discreatize(y, bounds=(-0.02, 0.02)):

    classes = [0, 1, 2]

    def apply(label):
        if label <= bounds[0]:
            return classes[0]
        elif label > bounds[1]:
            return classes[2]
        return classes[1]

    y_d = np.array(list(map(apply, y)))

    print(
        f'Class distributions: {((y_d == 0).sum(), (y_d == 1).sum(), (y_d == 2).sum())}')

    return y_d

# Check if label is 0-based
# base = np.min(y_train)
# if base != 0:
#     y_train -= base
# y_val -= base
