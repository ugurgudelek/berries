import pandas as pd
import os

print("Merging ...")
df = pd.DataFrame()
for filename in os.listdir():
    if filename.split('.')[-1] == 'csv':
        df = df.append(pd.read_csv(filename, index_col=0))
df.reset_index(inplace=True,drop=True)

# save to h5py
print("Creating h5py file")
df.to_hdf('dataholder.h5py',key='df')