import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../experiment/load_full_wo_feb29/result.csv', index_col=0, sep=';', header=1)

offset = 5
plt.scatter(list(range(96)), df.iloc[1000, offset:offset+96], label='y')
plt.scatter(list(range(96)), df.iloc[1000, offset+96:], label='yhat')
plt.scatter(list(range(96)), df.iloc[1050, offset:offset+96], label='y2')
plt.scatter(list(range(96)), df.iloc[1050, offset+96:], label='yhat2')
plt.legend()
plt.show()

print()