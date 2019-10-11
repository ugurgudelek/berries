__author__ = "Ugur Gudelek"
__email__ = "ugurgudelek@gmail.com"


"""
https://stackoverflow.com/questions/34764535/why-cant-matplotlib-plot-in-a-different-thread
"""

import matplotlib.pyplot as plt
import matplotlib
import threading
import time
import queue
import functools


#ript(Run In Plotting Thread) decorator
def ript(function):
    def ript_this(*args, **kwargs):
        global send_queue, return_queue, plot_thread
        if threading.currentThread() == plot_thread: #if called from the plotting thread -> execute
            return function(*args, **kwargs)
        else: #if called from a diffrent thread -> send function to queue
            send_queue.put(functools.partial(function, *args, **kwargs))
            return_parameters = return_queue.get(True) # blocking (wait for return value)
            return return_parameters
    return ript_this

#list functions in matplotlib you will use
functions_to_decorate = [[matplotlib.axes.Axes,'plot'],
                         [matplotlib.figure.Figure,'savefig'],
                         [matplotlib.backends.backend_tkagg.FigureCanvasTkAgg,'draw'],
                         ]
#add the decorator to the functions
for function in functions_to_decorate:
    setattr(function[0], function[1], ript(getattr(function[0], function[1])))

# function that checks the send_queue and executes any functions found
def update_figure(window, send_queue, return_queue):
    try:
        callback = send_queue.get(False)  # get function from queue, false=doesn't block
        return_parameters = callback() # run function from queue
        return_queue.put(return_parameters)
    except:
        None
    window.after(10, update_figure, window, send_queue, return_queue)

# function to start plot thread
def plot():
    # we use these global variables because we need to access them from within the decorator
    global plot_thread, send_queue, return_queue
    return_queue = queue.Queue()
    send_queue = queue.Queue()
    plot_thread=threading.currentThread()
    # we use these global variables because we need to access them from the main thread
    global ax, fig
    fig, ax = plt.subplots()
    # we need the matplotlib window in order to access the main loop
    window=plt.get_current_fig_manager().window
    # we use window.after to check the queue periodically
    window.after(10, update_figure, window, send_queue, return_queue)
    # we start the main loop with plt.plot()
    plt.show()


def main():
    #start the plot and open the window
    thread = threading.Thread(target=plot)
    thread.setDaemon(True)
    thread.start()
    time.sleep(1) #we need the other thread to set 'fig' and 'ax' before we continue
    #run the simulation and add things to the plot
    global ax, fig
    for i in range(10):
        ax.plot([1,i+1], [1,(i+1)**0.5])
        fig.canvas.draw()
        fig.savefig('updated_figure.png')
        time.sleep(1)
    print('Done')
    thread.join() #wait for user to close window

main()