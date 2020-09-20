from datetime import datetime, timedelta
from . import db

from .categories import importance

class ModelTypes:
    STUPID = "stupid"
    SIMPLE_STAT = "simple_stat"


class Model:
    def predict_categories(self, part: dict):
        raise NotImplementedError()

    def predict_week_amt(self, part: dict):
        raise NotImplementedError()


class StupidModel(Model):
    def predict_categories(self, part: dict):
        return ["Фаст Фуд", "Кино", "Одежда/Обувь"]

    def predict_week_amt(self, part: dict):
        return 6000


class SimpleStatModel(Model):
    def predict_categories(self, part: dict):
        now = part["today"]
        all_transactions = db.transactions.find(
            {
                "transaction_dttm": {"$gt": now - timedelta(days=7), "$lt": now},
                "party_rk": part["party_rk"]
            },
        )
        result = {}
        for tr in all_transactions:
            cat = tr["category"]
            result[cat] = result.setdefault(cat, 0) + tr["transaction_amt_rur"]
        with_importance_list = sorted([(k, v, v * importance[str(k)]) for k, v in result.items()], key=lambda x: -x[-1])
        lst = [x for x in (w[0] for w in with_importance_list) if isinstance(x, str) and x.lower() != "nan"]
        return lst[:3]

    def predict_week_amt(self, part: dict):
        now = part["today"]
        all_transactions = db.transactions.find(
            {
                "transaction_dttm": {"$gt": now - timedelta(days=7), "$lt": now},
                "party_rk": part["party_rk"],
                "category": part["curr_challenge_category"]
            },
        )
        return sum((t["transaction_amt_rur"] for t in all_transactions))


models_to_use = {
    ModelTypes.STUPID: StupidModel(),
    ModelTypes.SIMPLE_STAT: SimpleStatModel(),
}


def get_model(name: str, default=None) -> Model:
    return models_to_use.get(name, default)
