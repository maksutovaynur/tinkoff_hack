import os
import json
import pickle
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from datetime import datetime


# DATADIR = "/content/drive/My Drive/sk_tinkoff_hack/tinkoff_hack_data" # "./data"
transactions_path = f"{DATADIR}/avk_hackathon_data_transactions.csv"
pd.read_csv(f"{DATADIR}/avk_hackathon_data_transactions.csv")



############# avk_hackathon_data_party_x_socdem #############

df_socdem = pd.read_csv(f"{DATADIR}/avk_hackathon_data_party_x_socdem.csv")


def get_age_category(x):
  if x < 18:
    cat = 0
  elif x < 30:
    cat = 1
  elif x < 50:
    cat = 2
  else:
    cat = 3
  return cat


unk_token = "<UNK>"

df_socdem['age_category'] = df_socdem['age'].apply(lambda x: get_age_category(x))
df_socdem['gender_cd'] = df_socdem['gender_cd'].fillna(unk_token).astype(str)
df_socdem['marital_status_desc'] = df_socdem['marital_status_desc'].fillna(unk_token).astype(str)
df_socdem['children_indicator'] = df_socdem['children_cnt'].apply(lambda x: int(x > 0))


party_rk_label_to_idx = dict(zip(df_socdem['party_rk'].values, df_socdem['party_rk'].index))
party_rk_idx_to_label = dict(zip(df_socdem['party_rk'].index, df_socdem['party_rk'].values))



############# avk_hackathon_data_story_logs #############

df_story_logs = pd.read_csv(f"{DATADIR}/avk_hackathon_data_story_logs.csv")
df_story_logs = df_story_logs.rename(columns={"category": "story_category"})

most_liked_dct = dict(df_story_logs[df_story_logs['event'] == 'like'].groupby('party_rk')['story_category'].agg(lambda x:x.value_counts().index[0]))
most_favorite_dct = dict(df_story_logs[df_story_logs['event'] == 'favorite'].groupby('party_rk')['story_category'].agg(lambda x:x.value_counts().index[0]))
most_dislike_dct = dict(df_story_logs[df_story_logs['event'] == 'dislike'].groupby('party_rk')['story_category'].agg(lambda x:x.value_counts().index[0]))


def get_most_popular_category(x, popularity_dct):
  if x in popularity_dct:
    return popularity_dct[x]
  else:
    return "<UNK>"


df_story_logs['most_popular_like_category'] = df_story_logs['party_rk'].apply(lambda x: get_most_popular_category(x, most_liked_dct))
df_story_logs['most_popular_favorite_category'] = df_story_logs['party_rk'].apply(lambda x: get_most_popular_category(x, most_favorite_dct))
df_story_logs['most_popular_dislike_category'] = df_story_logs['party_rk'].apply(lambda x: get_most_popular_category(x, most_dislike_dct))

df_story_logs = df_story_logs.drop(columns=['date_time', 'story_id', 'story_category', 'event'])

party_rk_without_stories_lst = list(set(list(party_rk_label_to_idx.keys())) - set(df_story_logs['party_rk'].values))
party_rk_without_stories_df = pd.DataFrame(data=[[ind, "<UNK>", "<UNK>", "<UNK>"] for ind in party_rk_without_stories_lst],
                                           columns=['party_rk', 'most_popular_like_category',
                                                    'most_popular_favorite_category',
                                                    'most_popular_dislike_category'])

df_story_logs = df_story_logs.append(party_rk_without_stories_df)
df_story_logs = df_story_logs.drop_duplicates()



############# avk_hackathon_data_party_products #############

df_party_products = pd.read_csv(f"{DATADIR}/avk_hackathon_data_party_products.csv")
df_party_products['products_sum'] = df_party_products.drop(columns=['party_rk']).sum(axis=1)

most_popular_product_name = df_party_products.drop(columns=['party_rk', 'products_sum']).sum().idxmax()
df_party_products['most_popular_product_chosen'] = df_party_products[most_popular_product_name].apply(lambda x: int(x == 1))



############# avk_hackathon_data_transactions #############

df_transactions = pd.read_csv(f"{DATADIR}/avk_hackathon_data_transactions.csv")

def get_season(month_number):
  if month_number in [1,2,12]:
    return 0
  elif month_number in [3,4,5]:
    return 1
  elif month_number in [6,7,8]:
    return 2
  elif month_number in [9,10,11]:
    return 3


df_holidays = pd.read_csv(f"{DATADIR}/fcal_Russia_2019_2021.csv", encoding='cp1251', delimiter=";")
df_holidays['Date'] = df_holidays['Date'].apply(lambda x: '-'.join(x.split('.')[::-1]))
df_holidays = df_holidays[['Date', 'Designation']]
date_to_holiday_dct = dict(zip(df_holidays['Date'].values, df_holidays['Designation'].values))


df_transactions['year'] = df_transactions['transaction_dttm'].apply(lambda x: int(x.split('-')[0]))
df_transactions['month'] = df_transactions['transaction_dttm'].apply(lambda x: int(x.split('-')[1]))
df_transactions['day'] = df_transactions['transaction_dttm'].apply(lambda x: int(x.split('-')[2]))
df_transactions['season'] = df_transactions['month'].apply(lambda x: get_season(x))
df_transactions['log_transaction_amt_rur'] = df_transactions['transaction_amt_rur'].apply(lambda x: np.log(1 + x))


def get_holiday_name(x, date_to_holiday_dct=date_to_holiday_dct):
  if x in date_to_holiday_dct:
    return date_to_holiday_dct[x]
  else:
    return 'No holiday'


df_transactions['holiday_name'] = df_transactions['transaction_dttm'].apply(lambda x: get_holiday_name(x))
df_transactions['transaction_dttm_datetime'] = df_transactions['transaction_dttm'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
df_transactions['day_of_week'] = df_transactions['transaction_dttm_datetime'].dt.dayofweek

df_transactions = df_transactions.dropna(subset=['transaction_amt_rur', 'merchant_rk', 'merchant_type'])
df_transactions['merchant_group_rk'] = df_transactions['merchant_group_rk'].fillna(unk_token).astype(str)
df_transactions['category'] = df_transactions['category'].fillna(unk_token).astype(str)

df_transactions = df_transactions.drop(columns=['account_rk', 'transaction_amt_rur', 'transaction_dttm'])


# merge
df_transactions = df_transactions.merge(df_socdem, on='party_rk', how='inner')
df_transactions = df_transactions.rename(columns={"category": "transaction_category"})
df_transactions = df_transactions.merge(df_story_logs, on='party_rk', how='inner')
df_transactions = df_transactions.merge(df_party_products, on='party_rk', how='inner')
df_transactions = df_transactions.sort_values(by=['transaction_dttm_datetime'])
df_transactions.to_csv('overall_df_transactions.csv', index=False)