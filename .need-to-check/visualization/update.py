__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread
import time


def run_async(func):
    """
        run_async(func)
            function decorator, intended to make "func" run in a separate
            thread (asynchronously).
            Returns the created Thread object

            E.g.:
            @run_async
            def task1():
                do_something

            @run_async
            def task2():
                do_something_too

            t1 = task1()
            t2 = task2()
            ...
            t1.join()
            t2.join()
    """
    from threading import Thread
    from functools import wraps

    @wraps(func)
    def async_func(*args, **kwargs):
        func_hl = Thread(target=func, args=args, kwargs=kwargs)
        func_hl.start()
        return func_hl

    return async_func


def line_creator(x, phase):
    return np.sin(x + phase)


class Figure:
    def __init__(self):
        plt.ion()
        self.x = np.linspace(0, 6 * np.pi, 100)
        self.y = np.sin(self.x)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.traces = [
            {'line': self.ax.plot(self.x, self.y, 'r-')[0], 'data': line_creator},
            {'line': self.ax.plot(self.x, self.y, 'b*')[0], 'data': line_creator},
        ]

        func_hl = Thread(target=self.update, args=(self.fig, self.traces, self.x))
        func_hl.start()

    @staticmethod
    def update(fig, traces, x):
        for phase in np.linspace(0, 10 * np.pi, 500):
            for trace in traces:
                trace['line'].set_ydata(trace['data'](x, phase))
            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == "__main__":
    fig = Figure()
    print("update")
    time.sleep(2)
