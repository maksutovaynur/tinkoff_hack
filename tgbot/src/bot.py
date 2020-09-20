import telebot
from telebot.types import Message, ReplyKeyboardMarkup, KeyboardButton
from . import config as S
from . import db
from datetime import datetime, timedelta
from .models import get_model, ModelTypes

from .utils import log

bot = telebot.TeleBot(S.BOT_TOKEN)


class CTypes:
    ASSIGN = "/sign in as a customer"
    HELP = "/help"
    NEXT_DAY = "/move to next day"


class CParts:
    SPEND = "/spend"
    SELECT_CATEGORY = "/select"
    SELECT_PERCENT = "/by"


@bot.message_handler(content_types=["text"])
def root(message: Message):
    log("root handler")
    chat_id = message.from_user.id
    t = message.text
    if t == CTypes.ASSIGN:
        log(f"{CTypes.ASSIGN} message")
        Commands.assign(chat_id)
    elif t == CTypes.HELP:
        log(f"{CTypes.HELP} message")
        Commands.help(chat_id)
    else:
        part = Logics.get_exiting_party_rk(chat_id)
        if part is None:
            log(f"{CTypes.HELP} message")
            Commands.help(chat_id)
            return
        if t == CTypes.NEXT_DAY:
            Logics.next_day(part)
        elif t.startswith(CParts.SPEND):
            sum_str = t.rsplit(" ", 1)[-1]
            try: sum = float(sum_str)
            except:
                bot.send_message(chat_id, f"Please repeat: wrong sum '{sum_str}'")
                return
            Logics.spend(part, sum)
        elif t.startswith(CParts.SELECT_CATEGORY):
            Logics.select_category(part, t.split(" ", 1)[-1])
        elif t.startswith(CParts.SELECT_CATEGORY):
            Logics.select_category(part, t.split(" ", 1)[-1])
        else:
            bot.send_message(chat_id, "Sorry, unknown action")
            Commands.help(chat_id)


def create_keyboard(*rows):
    kb = ReplyKeyboardMarkup()
    for row in rows:
        kb.row(*[KeyboardButton(a) for a in row])
    return kb


root_kb = create_keyboard([CTypes.HELP, CTypes.ASSIGN])


class Commands:
    @classmethod
    def help(cls, chat_id):
        bot.send_message(
            chat_id,
            "Welcome to Tinkoff Save Adviser! Choose what you want to do:",
            reply_markup=root_kb
        )

    @classmethod
    def assign(cls, chat_id):
        d_find = {"chat_id": chat_id}
        result = db.app_state.find_one(d_find)
        if result is not None:
            bot.send_message(chat_id, "Deleted existing simulation data")
            db.app_state.delete_many(d_find)
        party_data = Logics.gen_random_party_rk()
        if party_data is None:
            bot.send_message(chat_id, "Internal error when trying to assign test data")
            return
        app_data = dict(
            **party_data,
            weekday_to_start=1,
            model_to_use=ModelTypes.STUPID,
            today=datetime.fromisoformat("2019-06-01"),
            chat_id=chat_id,
            account_money=30000,
            curr_challenge_start_dttm=None,
            curr_challenge_end_dttm=None,
            curr_challenge_category=None,
            curr_challenge_pred_amt=None,
            curr_challenge_percent=None,
            curr_challenge_spent_amt=None,
            challenge_history=[],
        )
        db.update_or_insert_app_data(app_data)
        bot.send_message(
            chat_id,
            f"Simulation started [your party_rk={app_data['party_rk']}]",
            reply_markup=create_keyboard()
        )
        Logics.suggest_new_categories(app_data)


class Logics:
    @classmethod
    def gen_random_party_rk(cls):
        while True:
            result = list(db.socdem.aggregate([{"$sample": {"size": 1}}]))
            log(f"Get Random RK result: {result}")
            if len(result) == 0:
                return
            result = result[0]
            existing = db.app_state.find_one({"party_rk": result["party_rk"]})
            if existing is None:
                return result

    @classmethod
    def get_exiting_party_rk(cls, chat_id):
        return db.app_state.find_one({"chat_id": chat_id})

    @classmethod
    def next_day(cls, part: dict):
        prev: datetime = part["today"]
        now = prev + timedelta(days=1)
        part["today"] = now
        cls.save_part(part)
        if now.weekday() == part["weekday_to_start"]:
            cls.summarize_past_period(part)
            cls.suggest_new_categories(part)

    @classmethod
    def summarize_past_period(cls, part):
        if part["curr_challenge_category"] is not None:
            amt_spent, amt_limit, amt_pred, percent = get_amts(part)
            cls.notify(
                part,
                f"Great! You made a {percent}% challenge! "
                f"You've spent {amt_spent} on '{part['category']}' (< {amt_pred})"
                if amt_spent < amt_limit else
                f"You could better! "
                f"You already spent less than last week ({amt_spent} over {amt_pred}), "
                f"but you did not reduce your spendings by {percent}%"
                if amt_spent < amt_pred else
                "Oh no! You failed last week's challenge!"
            )

    @classmethod
    def suggest_new_categories(cls, part: dict):
        model = get_model(part["model_to_use"])
        if model is None:
            cls.notify(part, f"Internal error: no such model: {part['model_to_use']}", show_status=False)
        categories_to_optimize = model.predict_categories(part)
        cls.notify(
            part,
            f"Based on your previous spendings, we found {len(categories_to_optimize)} categories you can optimize:",
            markup=create_keyboard([f"{CParts.SELECT_CATEGORY} {cat}" for cat in categories_to_optimize])
        )

    @classmethod
    def spend(cls, part: dict, sum: float):
        pass
    PERCENTAGES = [10, 25, 50]

    @classmethod
    def select_category(cls, part: dict, category):
        part["weekday_to_start"] = part["today"].weekday()
        part["curr_challenge_start_dttm"] = part["today"]
        part["curr_challenge_end_dttm"] = part["today"] + timedelta(days=7)
        part["curr_challenge_category"] = category
        part["curr_challenge_spent_amt"] = 0
        part["curr_challenge_percent"] = 10
        part["curr_challenge_pred_amt"] = get_model(part["model_to_use"])
        cls.save_part(part)
        cls.notify(
            part,
            "Fine! By how many percent you challenge to reduce your spendings?",
            markup=create_keyboard([f"{CParts.SELECT_PERCENT} {x}" for x in cls.PERCENTAGES])
        )

    @classmethod
    def select_percentage(cls, part: dict, percentage):
        part["curr_challenge_percent"] = percentage
        cls.notify(part, "Fine! Let's start saving your money.")

    @classmethod
    def todays_goal(cls, part):
        days_from_start = (part["today"] - part["curr_challenge_start_dttm"]).days
        cls.notify(part, f"Day {1 + days_from_start}")

    @classmethod
    def save_part(cls, part: dict):
        db.app_state.update_one(
            {"chat_id": part["chat_id"]},
            update={"$set": part},
            upsert=True
        )

    @classmethod
    def notify(cls, part, message, markup=None, show_status=True):
        if show_status: cls.status_message(part)
        bot.send_message(part["chat_id"], message, reply_markup=markup)

    @classmethod
    def status_message(cls, part):
        bot.send_message(
            part["chat_id"],
            f"Today: {part['today'].strftime('YYYY-mm-dd')}\n"
            + f"Current challenge category: {part['curr_challenge_category']}\n"
            if part['curr_challenge_category'] is not None else "no challenge\n"
            f"Account money left: {part['account_money']}\n"
        )


def get_amts(part: dict):
    pred_amt = part["curr_challenge_pred_amt"]
    spent_amt = part["curr_challenge_spent_amt"]
    percent = part["curr_challenge_percent"]
    limit = pred_amt * (100 - percent) / 100
    return spent_amt, limit, pred_amt, percent
