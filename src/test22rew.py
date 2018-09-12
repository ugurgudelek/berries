import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x1 = np.random.rand(101)
x2 = np.random.rand(101)
x1 = np.cumsum(x1)
x2 = np.cumsum(x2)

x1[:31] = -x1[:31]

plt.plot(x1, '-b', x2, '-r', np.diff(x1), '-g')


diff1 = np.diff(x1)
diff2 = np.diff(x2)

# print(np.sum(((diff1 * diff2) > 0)) / len(diff1))

a = [True, True, False]
print(np.sum([True, True, False]) / len(a))

