import telebot
from telebot.types import Message, ReplyKeyboardMarkup, KeyboardButton
from . import config as S
from . import db
from datetime import datetime

from .utils import log

bot = telebot.TeleBot(S.BOT_TOKEN)


class CmdTypes:
    START = "/start"
    ASSIGN = "/assign"
    CONTINUE = "/continue"
    FLUSH = "/flush"
    HELP = "/help"


@bot.message_handler(content_types=["text"])
def root(message: Message):
    log("root handler")
    chat_id = message.from_user.id

    if message.text == CmdTypes.ASSIGN:
        log(f"{CmdTypes.ASSIGN} message")
        Commands.assign(chat_id)
    else:
        part = get_exiting_party_rk(chat_id)
        if part is None:
            log(f"{CmdTypes.HELP} message")
            Commands.help(chat_id)
            return


def create_keyboard(*answers):
    kb = ReplyKeyboardMarkup()
    for a in answers:
        kb.add(KeyboardButton(a))
    return kb


class Commands:
    @classmethod
    def help(cls, chat_id):
        bot.send_message(
            chat_id,
            "Welcome to Tinkoff Save Adviser! Choose what you want to do:",
            reply_markup=create_keyboard(CmdTypes.HELP, CmdTypes.ASSIGN)
        )

    @classmethod
    def assign(cls, chat_id):
        d_find = {"chat_id": chat_id}
        result = db.app_state.find_one(d_find)
        if result is not None:
            bot.send_message(chat_id, "Deleted existing simulation data")
            db.app_state.delete_one(d_find)
        party_data = gen_random_party_rk()
        if party_data is None:
            bot.send_message(chat_id, "Internal error when trying to assign test data")
            return
        app_data = dict(
            **party_data,
            today=datetime.fromisoformat("2019-06-01"),
            chat_id=chat_id,
            curr_challenge_start_dttm=None,
            curr_challenge_end_dttm=None,
            curr_challenge_category=None,
            curr_challenge_pred_amt=None,
            curr_challenge_percent=None,
            challenge_history=[],
        )
        db.update_or_insert_app_data(app_data)
        bot.send_message(chat_id, f"Simulation started [your party_rk={app_data['party_rk']}]")

    @classmethod
    def flush(cls, chat_id):
        d_find = {"chat_id": chat_id}
        result = db.app_state.find_one(d_find)
        if result is not None:
            bot.send_message(chat_id, "Deleted existing simulation data")
            db.app_state.delete_one(d_find)


def gen_random_party_rk():
    while True:
        result = list(db.socdem.aggregate([{"$sample": {"size": 1}}]))
        log(f"Get Random RK result: {result}")
        if len(result) == 0:
            return
        result = result[0]
        existing = db.app_state.find_one({"party_rk": result["party_rk"]})
        if existing is None:
            return result


def get_exiting_party_rk(chat_id):
    return db.app_state.find_one({"chat_id": chat_id})
