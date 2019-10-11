import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def random_walk(origin, step_n, plot=False):
    # Define parameters for the walk
    dims = 1

    step_set = [-1, 0, 1]
    origin = np.zeros((1,dims)) + origin
    # Simulate steps in 1D
    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape)
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]

    if plot:
        # Plot the path
        fig = plt.figure(figsize=(8,4),dpi=200)
        ax = fig.add_subplot(111)
        ax.scatter(np.arange(step_n+1), path, c='blue',alpha=0.25,s=0.05);
        ax.plot(path,c='blue',alpha=0.5,lw=0.5,ls='-',);
        ax.plot(0, start, c='red', marker='+')
        ax.plot(step_n, stop, c='black', marker='o')
        plt.title('1D Random Walk')
        plt.tight_layout(pad=0)
        plt.show()
        plt.savefig('random_walk_1d.png',dpi=250);

    return path[:-1, 0]

df = pd.read_csv('spy.csv')
step_n = df.shape[0]
step_n
df.head()
df.loc[0, ['low']]

df['low'] = random_walk(df.loc[0, 'low'], step_n)
df['high'] = random_walk(df.loc[0, 'high'], step_n)
df['open'] = random_walk(df.loc[0, 'open'], step_n)
df['close'] = random_walk(df.loc[0, 'close'], step_n)
df['adjusted_close'] = df['close']

df.to_csv('synthetic_random.csv', index=False)
