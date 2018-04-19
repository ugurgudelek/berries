
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """

    """

    def __init__(self):
        self.container = dict()

        #todo - generalize
        self.container['tloss'] = np.array([])
        self.container['vloss'] = np.array([])

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.show(block=False)



    def append_data(self, name, data):
        self.container[name] = np.append(self.container[name], data)

    def visualize(self):
        self.ax.clear()

        for key, value in self.container.items():
            plt.plot(value, label=key)

        plt.legend()
        plt.pause(0.01)
        self.fig.canvas.draw()


def display_errors(history):

    t_error = []
    for epoch, epoch_history in history.items():
        t_error.append(epoch_history['train']['loss'].mean())

    print(t_error[-1])
    plt.plot(np.log(t_error))

    fig.canvas.draw()
    plt.pause(0.0001)