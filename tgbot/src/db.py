from . import config as S
import pymongo as __pm


__client = __pm.MongoClient(host=S.MONGO_HOST, port=S.MONGO_PORT)
__app_database = __client.get_database("tinkoff-bot")

__existing_data = __client.get_database("tinkoff-data")


transactions = __existing_data["transactions"]
balances = __existing_data["balances"]
socdem = __existing_data["socdem"]


app_state = __app_database["state"]


__fields = [
    "chat_id",
    "party_rk",
    "today",
    "curr_challenge_start_dttm",
    "curr_challenge_end_dttm",
    "curr_challenge_category",
    "curr_challenge_pred_amt",
    "curr_challenge_percent",
    "challenge_history"
]


def update_or_insert_app_data(data):
    app_state.update_one({"party_rk": data["party_rk"]}, update={"$set": data}, upsert=True)
