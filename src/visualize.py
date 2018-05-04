
import matplotlib.pyplot as plt
import numpy as np
import io
import imageio

class Visualizer:
    """

    """

    def __init__(self):
        self.container = dict()

        #todo - generalize
        self.container['tloss'] = np.array([])
        self.container['vloss'] = np.array([])

        # self.fig = plt.figure()
        # self.ax = self.fig.add_subplot(1, 1, 1)
        # todo- fix xlim. 28.04.2018
        # plt.xlim((0,50))
        # plt.ylim((0,1))
        # plt.show(block=False)

    def prediction_to_image(self, actual, prediction, im_title):

        # ax.clear()
        plt.figure()
        plt.plot(actual, label='real')
        plt.plot(prediction, label='pred')
        plt.legend()
        plt.suptitle('{}'.format(im_title))
        plt.legend()
        # plt.pause(1)
        # fig.canvas.draw()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = imageio.imread(buf)
        buf.close()
        return im

    def append_data(self, name, data):
        self.container[name] = np.append(self.container[name], data)

    def visualize(self, epoch):
        self.ax.clear()

        for key, value in self.container.items():
            plt.scatter(range(len(value)), value, label=key)

        self.ax.text(3, 8, 'cem', style='italic',
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        plt.legend()
        plt.pause(1)
        self.fig.canvas.draw()

    def report(self):
        for key,value in self.container.items():
            print('key:{} - value:{}'.format(key, value))




class Plotter:
    """
    Visualizes all related info.
    """

    def __init__(self, xlim, ylim=(0, 1), block=False):

        self.xlim = xlim
        self.ylim = ylim
        self.block = block

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.show(block=block)

        self.plot_container = defaultdict(tuple)

    def plot(self):
        """
        Plot all appended figures.

        Returns:

        """
        self.ax.clear()
        x = np.arange(0, self.xlim[1])
        for (label, (plot_type, what_to_plot)) in self.plot_container.items():
            if plot_type == 'line':
                # TODO: need to refactor this method!! It's to strict
                if label == 'X':
                    plt.plot(x[:-1], what_to_plot[1:], label='X')
                elif label == 'true':
                    last_value_X = self.plot_container['X'][1][-1]
                    plt.plot(x[-2:], np.stack((last_value_X, what_to_plot)), label='true')
                elif label == 'pred':
                    last_value_X = self.plot_container['X'][1][-1]
                    plt.plot(x[-2:], np.stack((last_value_X, what_to_plot)), label='pred')
                else:
                    plt.plot(what_to_plot[-self.xlim[1]:], label=label)
            if plot_type == 'scatter':
                plt.scatter(x=list(range(self.xlim[1])), y=what_to_plot[-self.xlim[1]:], label=label)

        plt.legend()
        plt.pause(0.0001)
        self.fig.canvas.draw()

    def add(self, what_to_plot, plot_type, label):
        """
        Add what_to_plot to plot container.
        Args:
            what_to_plot:
            plot_type:
            label:

        Returns:

        """
        self.plot_container[label] = (plot_type, what_to_plot)

    def drop(self, label):
        """
        Drop label from plot container.
        Args:
            label:

        Returns:

        Raises: KeyError

        """
        self.plot_container.pop(label)


def display_errors(history):

    t_error = []
    for epoch, epoch_history in history.items():
        t_error.append(epoch_history['train']['loss'].mean())

    print(t_error[-1])
    plt.plot(np.log(t_error))

    fig.canvas.draw()
    plt.pause(0.0001)