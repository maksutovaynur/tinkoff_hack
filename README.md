# 2x3 team project to the Tinkoff Hack

## Project goals

### Problem

People spend more money than they really need. There are necessary expenses that you can not avoid.
But there are categories of expences that can be reduced.

### Solution in general

We provide customer with the solution, that helps him identify least necessary expenses 
and by everyday communication allows him reduce them.

### Detailed solution

There are several parts of the system

1) Categories evaluator. This is a model that, based on previous transactions, can suggest, 
which expenses can be counted as unnecessary. The final choice is given to the client, 
we just give an advice.

2) Week scheduler. This block plans how much money can spend each and every day on monitored category,
and analyses everyday changes on account balance and transactions.

3) Notifier/Gamificator. Based on information from the scheduler, it develops helpful advices, 
motivates customer by challenging him and showing "what if". 
It also reports weekly and monthly statistics about saved money and gives game achievements. 


### Additional work

We also improved baseline model solution, provided by the Tinkoff team.


## Project structure

### 1) [Telegram Bot Prototype](tgbot/README.md)
### 2) [Baseline model improvement](models/README.md)
### 3) [Feature extraction for models](data_processing/README.md)

