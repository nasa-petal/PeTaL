import pandas as pd

df = pd.read_hdf('storage.h5', 'data')
print(df)
print(df.columns)
