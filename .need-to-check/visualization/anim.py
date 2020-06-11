import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random

def list_gen():
    return [random.random() for _ in range(100)]

def plot_cont(fun, xmax):

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def update(i):
        y=list_gen()
        x = range(len(y))
        ax.clear()
        ax.plot(x, y)

    a = anim.FuncAnimation(fig, update, frames=xmax, repeat=True)
    plt.show()

plot_cont(lambda :random.randint(1,100), 10)
print("sdfs")