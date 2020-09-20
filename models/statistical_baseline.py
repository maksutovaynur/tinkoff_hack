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


"""
A simple statistical baseline to calculate top-N merchant types in different groups
and then sort the results according to frequency.
"""

df_transactions = pd.read_csv('overall_df_transactions.csv')

top_five_df = df_transactions[['party_rk']].drop_duplicates()
top_five_df = top_five_df.sort_values(by='party_rk').reset_index().drop(columns='index')


def top_N_everyone_all_time(df, N=30):
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_everyone_winter(df, N=30):
  df = df[df['month'].apply(lambda x: x in [12,1,2])]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_everyone_new_year(df, N=30):
  df = df[df['month'].apply(lambda x: x == 1) & df['day'].apply(lambda x: x <= 8)]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_everyone_jan_feb(df, N=30):
  df = df[df['month'].apply(lambda x: x in [1,2])]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_everyone_region_flg(df, N=30):
  uniq_values_lst = np.unique(df['region_flg'].values)
  dct = {}
  for value in uniq_values_lst:
    dct[value] = df[df['region_flg'] == value]['merchant_type'].value_counts().index.tolist()[:N]
  return dct


def top_N_everyone_gender_cd(df, N=30):
  uniq_values_lst = np.unique(df['gender_cd'].values)
  dct = {}
  for value in uniq_values_lst:
    dct[value] = df[df['gender_cd'] == value]['merchant_type'].value_counts().index.tolist()[:N]
  return dct


def top_N_everyone_gender_cd_january(df, N=30):
  df = df[df['month'].apply(lambda x: x == 1)]
  uniq_values_lst = np.unique(df['gender_cd'].values)
  dct = {}
  for value in uniq_values_lst:
    dct[value] = df[df['gender_cd'] == value]['merchant_type'].value_counts().index.tolist()[:N]
  return dct


def top_N_everyone_age_category_winter(df, N=30):
  df = df[df['month'].apply(lambda x: x in [12,1,2])]
  uniq_values_lst = np.unique(df['age_category'].values)
  dct = {}
  for value in uniq_values_lst:
    dct[value] = df[df['age_category'] == value]['merchant_type'].value_counts().index.tolist()[:N]
  return dct


def top_N_everyone_children_indicator_winter(df, N=30):
  df = df[df['month'].apply(lambda x: x in [12,1,2])]
  uniq_values_lst = np.unique(df['children_indicator'].values)
  dct = {}
  for value in uniq_values_lst:
    dct[value] = df[df['children_indicator'] == value]['merchant_type'].value_counts().index.tolist()[:N]
  return dct


def top_N_everyone_financial_account_type_cd_winter(df, N=30):
  df = df[df['month'].apply(lambda x: x in [12,1,2])]
  uniq_values_lst = np.unique(df['financial_account_type_cd'].values)
  dct = {}
  for value in uniq_values_lst:
    dct[value] = df[df['financial_account_type_cd'] == value]['merchant_type'].value_counts().index.tolist()[:N]
  return dct


def top_N_party_rk_all_time(party_rk, df=df_transactions, N=30):
  df = df[df['party_rk'] == party_rk]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_party_rk_last_week(party_rk, df=df_transactions, N=30):
  df = df[df['party_rk'] == party_rk]
  df = df[df['month'].apply(lambda x: x == 12)]
  df = df[df['day'].apply(lambda x: x >= 24)]
  if df.shape[0] > 0:
    lst = df['merchant_type'].value_counts().index.tolist()[:N]
  else:
    lst = []
  return lst


def top_N_party_rk_pokupka(party_rk, df=df_transactions, N=30):
  df = df[df['party_rk'] == party_rk]
  df = df[df['transaction_type_desc'].apply(lambda x: x == 'Покупка')]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_party_rk_last_month(party_rk, df=df_transactions, N=30):
  df = df[df['party_rk'] == party_rk]
  df = df[df['year'].apply(lambda x: x == 2019)]
  df = df[df['month'].apply(lambda x: x == 12)]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_party_rk_new_year(party_rk, df=df_transactions, N=30):
  df = df[df['party_rk'] == party_rk]
  df = df[df['month'].apply(lambda x: x == 1) & df['day'].apply(lambda x: x <= 8)]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_party_rk_winter(party_rk, df=df_transactions, N=30):
  df = df[df['party_rk'] == party_rk]
  df = df[df['month'].apply(lambda x: x in [12,1,2])]
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_party_rk_holidays(party_rk, df=df_transactions, N=30):
  df = df[df['party_rk'] == party_rk]
  df = df[df['holiday_name'] != 'No holiday']
  lst = df['merchant_type'].value_counts().index.tolist()[:N]
  return lst


def top_N_everyone_gender_cd_plus_age_category(df, N=30):
  gender_cd_uniq_lst = np.unique(df['gender_cd'].values)
  age_category_uniq_lst = np.unique(df['age_category'].values)

  dct = {}
  for gender in gender_cd_uniq_lst:
    for age_cat in age_category_uniq_lst:
      small_df = df[df['gender_cd'].apply(lambda x: x == gender)]
      small_df = df[df['age_category'].apply(lambda x: x == age_cat)]
      dct[(gender, age_cat)] = small_df['merchant_type'].value_counts().index.tolist()[:N]
  return dct


party_rk_lst = top_five_df['party_rk'].values

lst1 = top_N_everyone_all_time(df_transactions)
lst2 = top_N_everyone_winter(df_transactions)
lst3 = top_N_everyone_new_year(df_transactions)
lst4 = top_N_everyone_jan_feb(df_transactions)

dct1 = top_N_everyone_region_flg(df_transactions)
dct2 = top_N_everyone_gender_cd(df_transactions)
dct3 = top_N_everyone_gender_cd_january(df_transactions)
dct4 = top_N_everyone_age_category_winter(df_transactions)
dct5 = top_N_everyone_children_indicator_winter(df_transactions)
dct6 = top_N_everyone_financial_account_type_cd_winter(df_transactions)

dct7 = top_N_everyone_gender_cd_plus_age_category(df_transactions)

from collections import Counter
from tqdm.notebook import tqdm


output_dct = {}

for party_rk in tqdm(party_rk_lst):
    # print(party_rk)
    party_rk_data = df_transactions[df_transactions['party_rk'] == party_rk].iloc[0]
    counter_party_rk = Counter()


    counter_party_rk.update(lst1)
    counter_party_rk.update(lst2)
    counter_party_rk.update(lst3)
    counter_party_rk.update(lst4)


    counter_party_rk.update(dct1[party_rk_data['region_flg']])
    counter_party_rk.update(dct2[party_rk_data['gender_cd']])
    counter_party_rk.update(dct3[party_rk_data['gender_cd']])
    counter_party_rk.update(dct4[party_rk_data['age_category']])
    counter_party_rk.update(dct5[party_rk_data['children_indicator']])
    counter_party_rk.update(dct6[party_rk_data['financial_account_type_cd']])
    counter_party_rk.update(dct7[(party_rk_data['gender_cd'],
                                  party_rk_data['age_category'])])


    counter_party_rk.update(top_N_party_rk_all_time(party_rk))
    counter_party_rk.update(top_N_party_rk_last_week(party_rk))
    counter_party_rk.update(top_N_party_rk_last_month(party_rk))
    counter_party_rk.update(top_N_party_rk_new_year(party_rk))
    counter_party_rk.update(top_N_party_rk_winter(party_rk))
    counter_party_rk.update(top_N_party_rk_holidays(party_rk))
    counter_party_rk.update(top_N_party_rk_pokupka(party_rk))

    top30 = sorted(counter_party_rk, key=counter_party_rk.__getitem__)[::-1]
    output_dct[party_rk] = top30