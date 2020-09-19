import pandas as pd

def read_csvs(*filenames, **kwargs):
    dfs = (pd.read_csv(filename, **kwargs) for filename in filenames)
    return pd.concat(dfs, ignore_index=True)