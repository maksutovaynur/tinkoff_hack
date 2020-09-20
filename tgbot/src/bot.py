import telebot
from telebot.types import Message
from . import config as S
from . import db
bot = telebot.TeleBot(S.BOT_TOKEN)


class CmdTypes:
    START = "/start"
    ASSIGN = "/assign"
    FLUSH = "/flush"
    HELP = "/help"


@bot.message_handler(content_types=["text"])
def start(message: Message):
    if message.text == CmdTypes.START:
        Commands.help(message.from_user.id)
        bot.register_next_step_handler(message, root)


def root(message: Message):
    if message.text == CmdTypes.HELP:
        Commands.help(message.from_user.id)
    elif message.text == CmdTypes.ASSIGN:
        Commands.assign(message.from_user.id)


class Commands:
    @classmethod
    def help(cls, chat_id):
        bot.send_message(chat_id, "Type /reg to start simulation")

    @classmethod
    def assign(cls, chat_id):
        d_find = {"chat_id": chat_id}
        result = db.app_state.find_one(d_find)
        if result is not None:
            bot.send_message(chat_id, "Deleted existing simulation data")
            db.app_state.delete_one(d_find)
        get_random_party_rk()


    @classmethod
    def flush(cls, chat_id):
        d_find = {"chat_id": chat_id}
        result = db.app_state.find_one(d_find)
        if result is not None:
            bot.send_message(chat_id, "Deleted existing simulation data")
            db.app_state.delete_one(d_find)


def get_random_party_rk():
    result = db.socdem.aggregate([{"$sample": {"size": 1}}])
    print(f"Get Random RK result: {result}")
