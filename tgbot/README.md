# Telegram Bot Prototype

## What is this

We tried to show UX features we would like to have in the final product.
As the final product should be incorporated into existing Tinkoff banking application, 
we found cheap way to show UX, not trying to implement existing Tinkoff interface by ourselves.

This solution uses simplest statistical models to reveal "unnecessary" categories,
where customer can save more money. Our app prototype offers customer a challenge.
After accepting the challenge, game begins.

Prototype shows everyday support with tips and advices, reminds everyday spending limit on selected category.
After game round (7-days long) app summarizes your transaction activity and rewards your achivements.

Then, next round can be started again.


## How to run

1) Install docker and docker-compose
    
    https://docs.docker.com/compose/install/

2) Create new telegram bot. Keep its `api_key`

    https://core.telegram.org/bots

3) Clone this repo

     ```git clone https://github.com/maksutovaynur/tinkoff_hack.git```
    
4) Copy config file

    ```cp tgbot/src/config.example.py tgbot/config.py```
    
    and but your token there
    
    ```BOT_TOKEN = "your_api_key"```
    
5) Run docker-compose
    
    ```cd tgbot && docker-compose up --build -d```
    
6) After that you can communicate bot, using Telegram.


## Demo

1) join already running bot `@tinkoff_save_advisor_bot` bot     
2) see [screenshot](screenshot.png)