__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"


def run_async_thread(func):
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


import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from numpy import arange, sin, pi
from threading import Thread
from multiprocessing import Process
import time

# plt.ion()
# plt.show(block=False)

class Dummy():

    def plotme(self, iteration = 1):
        while True :
            print("%ix plotting... " % iteration)
            t = arange(0.0, 2.0, 0.01)
            s = sin(2*pi*t)

            fig, ax = plt.subplots()
            ax.plot(t, s)
            ax.set_xlabel('time (s)')
            ax.set_ylabel('voltage (mV)')
            ax.set_title('About as simple as it gets, folks (%i)' % iteration)
            # fig.savefig("19110942_%i_test.png" % iteration)
            plt.show()
            time.sleep(5)

    def threadme(self, iteration = 1):

        thread_plot = Process(target=self.plotme,
                                       args=(iteration,))
        thread_plot.start()
        thread_plot.join()


if __name__ == "__main__":
    dummy = Dummy()
    dummy.threadme(1)
    print("Hey")
    dummy.threadme(2)