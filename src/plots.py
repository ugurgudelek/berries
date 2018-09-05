from dataset import IndicatorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def point2label(data, on, window=15):
    """
    Finds turning points while iterating over data[on]

    Args:
        data: (pd.DataFrame) applied dataset
        on: applied column name
        window: rolling window size

    Returns: (pd.Series) labels

    """
    data = data.copy()
    data['maxs'] = data[on].rolling(window, center=True, min_periods=window).apply(
        IndicatorDataset.is_center_max)
    data['mins'] = data[on].rolling(window, center=True, min_periods=window).apply(
        IndicatorDataset.is_center_min)

    data['label'] = 'mid'
    data.loc[data['maxs'] == 1, 'label'] = 'top'
    data.loc[data['mins'] == 1, 'label'] = 'bot'

    data = data.drop(['maxs', 'mins'], axis=1)

    return data['label']


def filter_consequtive_same_label(labels):
    """

    Args:
        labels: (pd.Series)

    Returns:

    """
    state = None
    for i, label in enumerate(labels):
        if label == 'mid':
            continue
        if state is None:
            state = label
            continue
        if state == label:
            labels[i] = 'mid'
        else:
            state = label

    return labels


def crop_firstnonbot_and_lastnontop(labels):
    """

    Args:
        labels: (pd.Series)

    Returns:( pd.Series)

    """

    first_bot_idx = labels[(labels == 'bot')].index.values[0]
    last_top_idx = labels[(labels == 'top')].index.values[-1]

    return labels[first_bot_idx:last_top_idx + 1]


def plot_data_zigzag_tpoints(data, point_labels, zigzag_distances, label, colormap=None, linestyles=None):
    """
    Plots data & zigzag data & turning points

    Args:
        data: (pd.Series)
        labels: (pd.Series)
        label: (str) label of plot
        colormap: (dict) e.g : {'mid': (0.2, 0.4, 0.6, 0), 'top': (1, 0, 0, 0.7), 'bot': (0, 1, 0, 0.7)}
        linestyle: (str) e.g : --b

    Returns: fig

    """

    x = range(len(point_labels))

    # (r,g,b,a)
    if colormap is None:
        colormap = {'mid': (0.2, 0.4, 0.6, 0), 'top': (1, 0, 0, 0.7), 'bot': (0, 0, 1, 0.7)}
    if linestyles is None:
        linestyles = {'data': '--g', 'zigzag': '--y'}

    plt.scatter(x=x, y=data, c=[colormap[label] for label in point_labels], label=None, marker='^')
    plt.plot(x, data, linestyles['data'], alpha=0.6, label=label)
    plt.plot(x, zigzag_distances, linestyles['zigzag'], label='zigzag', alpha=0.6)
    plt.legend()

    return plt


def label2distance(data, labels):
    # point turning points then process for distance
    # after this line, stocks has 'label' column which has top-mid-bot values.

    def distance(idxs, turning_points):
        """
        Assumes turning_points start with increasing segment.
        Args:
            idxs: used for creation of dist array
            turning_points:

        Returns:

        """
        segments = np.array(list(zip(turning_points[:-1], turning_points[1:])))

        def calc_dist(data, current_ix, lower_ix, upper_ix):
            delta_x = upper_ix - lower_ix
            delta_y = data[upper_ix] - data[lower_ix]
            slope = delta_y / delta_x

            d = data[lower_ix] + slope * (current_ix - lower_ix)
            return d, slope

        dist = np.zeros_like(idxs, dtype=np.float)
        for (lower_ix, upper_ix) in segments:
            for current_ix in range(lower_ix, upper_ix):
                dist[current_ix], slope = calc_dist(data, current_ix, lower_ix, upper_ix)


        return dist

    mid_idxs = labels[labels == 'mid'].index.values
    top_idxs = labels[labels == 'top'].index.values
    bot_idxs = labels[labels == 'bot'].index.values

    turning_points = np.sort(np.concatenate((bot_idxs, top_idxs)))
    distances = distance(labels.index.values, turning_points)

    # at this point "label" values are between 0-1 range.
    # let me add -0.5 bias
    # distances = distances - 0.5

    return distances


def zigzag(stock, on, window):
    labels = point2label(stock, on=on, window=window)
    labels = filter_consequtive_same_label(labels=labels)
    labels = crop_firstnonbot_and_lastnontop(labels=labels)

    stock['label'] = labels
    stock = stock.dropna(axis=0).reset_index(drop=True)

    zigzag_distances = label2distance(stock[on], stock['label'])

    stock['zigzag'] = zigzag_distances

    return stock






stock = pd.read_csv('../input/spy.csv')
stock['pct_change'] = stock['adjusted_close'].pct_change()
stock = stock.dropna(axis=0).reset_index(drop=True)

stock = zigzag(stock, on='pct_change', window=15)

fig = plt.figure()

plot_data_zigzag_tpoints(data=stock['pct_change'],
                         point_labels=stock['label'],
                         zigzag_distances=stock['zigzag'],
                         linestyles={'data':'-r', 'zigzag':'--y'},
                         label='pct_change')
plt.plot(stock['close']/1500 - 0.2, label='close')
plt.show()


print()

