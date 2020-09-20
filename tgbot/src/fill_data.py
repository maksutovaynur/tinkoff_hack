import pandas as pd
from . import db
from . import config as S
from os.path import join


def insert_or_create(col, row):
    data = dict(**row._asdict())
    data["Index"] = f"{i}_{data['Index']}"
    col.update_one(
        {"Index": data["Index"]},
        update={"$set": data},
        upsert=True
    )


rows_written = 0
_MAX_PARTS = 12
_LOG_EVERY = 500


print("Balance info")
df = pd.read_csv(join(S.ROOT_DATA_FOLDER, "avk_hackathon_data_account_x_balance.csv"))
for row in df.itertuples():
    insert_or_create(db.balances, row)
    rows_written += 1
    if rows_written % _LOG_EVERY == 0: print(f"{rows_written} rows written")


print("Socdem info")
df = pd.read_csv(join(S.ROOT_DATA_FOLDER, "avk_hackathon_data_party_x_socdem.csv"))
for row in df.itertuples():
    insert_or_create(db.socdem, row)
    rows_written += 1
    if rows_written % _LOG_EVERY == 0: print(f"{rows_written} rows written")


print("Start write transactions")
for i, fn in enumerate(join(S.ROOT_DATA_FOLDER, f"avk_hackathon_data_transactions-{i}.csv") for i in range(1, 1 + _MAX_PARTS)):
    df = pd.read_csv(fn, parse_dates=["transaction_dttm"])
    for row in df.itertuples():
        insert_or_create(db.transactions, row)
        rows_written += 1
        if rows_written % _LOG_EVERY == 0: print(f"{rows_written} rows written")




