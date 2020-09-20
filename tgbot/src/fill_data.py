import pandas as pd
from . import db
from . import config as S
from os.path import join
import gc
from .utils import log


def preprocess_docs(row, idx):
    data = dict(**row._asdict())
    data["Index"] = f"{idx}_{data['Index']}"
    return data


def insert_or_create(col, rows):
    col.insert_many(rows)


rows_written = 0


def add_row(col, row, idx=0, write_each=2000, cache=[]):
    global rows_written
    if row is not None:
        cache.append(preprocess_docs(row, idx))
        rows_written += 1
    if len(cache) >= write_each or (row is None and len(cache) > 0):
        insert_or_create(col, cache)
        cache.clear()
        log(f"{rows_written} rows written")



_MAX_PARTS = 12


db.socdem.delete_many({})
db.balances.delete_many({})
db.transactions.delete_many({})


gc.collect()
log("Socdem info")
df = pd.read_csv(join(S.ROOT_DATA_FOLDER, "avk_hackathon_data_party_x_socdem.csv"))
for row in df.itertuples():
    add_row(db.socdem, row)
add_row(db.socdem, None)


gc.collect()
log("Balance info")
df = pd.read_csv(join(S.ROOT_DATA_FOLDER, "avk_hackathon_data_account_x_balance.csv"))
for row in df.itertuples():
    add_row(db.balances, row)
add_row(db.balances, None)


log("Start write transactions")
for i, fn in enumerate(join(S.ROOT_DATA_FOLDER, f"avk_hackathon_data_transactions-{j}.csv") for j in range(1, 1 + _MAX_PARTS)):
    gc.collect()
    df = pd.read_csv(fn, parse_dates=["transaction_dttm"])
    df["weeknum"] = df["transaction_dttm"].map(lambda x: x.isocalendar()[1])
    for row in df.itertuples():
        add_row(db.transactions, row)
    add_row(db.transactions, None)

