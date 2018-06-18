import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def merge_stocks(input_path, output_path):
    filelist = os.listdir(input_path)
    stocks = dict()

    for file in filelist:
        stockname, extension = os.path.splitext(file)
        if extension == '.csv':
            stockdata = pd.read_csv(os.path.join(input_path, file))
            stockdata['name'] = stockname
            stocks[stockname] = stockdata

    stock_dataframe = pd.DataFrame()
    for stockname, stockdata in stocks.items():
        stock_dataframe = pd.concat((stock_dataframe, stockdata))

    stock_dataframe.to_csv(os.path.join(output_path, 'stocks.csv'), index=False)

def roll_is_max(x):
    return True if x.max() == x[0] else False

def roll_is_min(x):
    return True if x.min() == x[0] else False




def pick_random_samples(df, on, condition, n):
    return df.loc[df[on] == condition].sample(n=n, replace=True)

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


# save some column
def save_column(stocks, col_name='adjusted_close'):
    def inner_func(data):
        data['raw_{}'.format(col_name)] = data[col_name].values
        return data

    return stocks.groupby('name').apply(inner_func)

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal

    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.


    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.

    Notes
    -----

    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.

    Examples
    --------

    >>> a = numpy.array([1, 2, 3, 4, 5])
    >>> sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    >>> sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])

    See Also
    --------
    pieces : Calculate number of pieces available by sliding

    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided

def plot_wrt_labels(dataset, timeseries_col, condition_func, text, arrow_len=20):

    timeseries_data = dataset[timeseries_col]
    arrow_positions = dataset.loc[dataset.apply(condition_func, axis=1)].index.values

    plt.plot(timeseries_data, label='close', c='r')

    # plt.axvline(200)
    # plt.axvline(400)
    # plt.axhline(100)
    # plt.axhline(150)

    for arrow_pos in arrow_positions:
        plt.annotate(text, xy=(arrow_pos, timeseries_data.iloc[arrow_pos]), xycoords='data',
                     xytext=(arrow_pos, timeseries_data.iloc[arrow_pos]+arrow_len), textcoords='data', fontsize=7,
                     color='#303030', arrowprops=dict(edgecolor='black', arrowstyle='->', connectionstyle="arc3"))
    # pylab.arrow(x=10,y=10,dx=100, dy=100, fc="k", ec="k", head_width=0.05, head_length=0.1 )


# input_path = '../dataset/finance/stocks/raw_stocks'
# output_path = '../dataset/finance/stocks'
# merge_stocks(input_path=input_path, output_path=output_path)

if __name__ == "__main__":
    df = pd.read_csv('../dataset/finance/stocks/indicator_dataset.csv')

    plot_wrt_labels(df, timeseries_col='raw_adjusted_close', condition_func=lambda row:row['label_sell'] == 1, text='sell', arrow_len=10)
    plot_wrt_labels(df, timeseries_col='raw_adjusted_close', condition_func=lambda row:row['label_buy'] == 1, text='buy', arrow_len=10)
    plt.legend()
    plt.show()