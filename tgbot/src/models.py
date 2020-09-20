from datetime import datetime


class ModelTypes:
    STUPID = "stupid"


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


models_to_use = {
    ModelTypes.STUPID: StupidModel()
}


def get_model(name: str, default=None) -> Model:
    return models_to_use.get(name, default)
