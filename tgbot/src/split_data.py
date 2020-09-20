import pandas as pd
import os
import sys


file_name = os.path.abspath(os.path.expanduser(sys.argv[1]))
base_name = file_name.rsplit(".", 1)[0]
data = pd.read_csv(file_name)

max_size = int(sys.argv[2])
for i, line in enumerate(range(0, data.shape[0], max_size)):
    subdata = data.iloc[line: line + max_size]
    subdata.to_csv(f"{base_name}-{i + 1}.csv")

