import json
import pickle
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def prepare_data(party_list, mode="train"):
    """
    This function define the pipeline of the creation of train and valid samples.
    We consider each client from party_list. For each client take each
    predict_period_start from predict_dates list. All client transaction before
    this date is our features. Next, we look at the customer's transactions in
    the next two months. This transactions should be predicted. It will form
    our labels vector.
    """

    data_sum = []
    data_trans_type = []
    data_merchant_type = []
    data_labels = []

    data_fin_type = []
    data_trans_cat = []
    data_day = []
    data_gender = []
    data_age = []
    data_marital_status = []
    data_child_cnt = []
    data_like_cat = []
    data_fav_cat = []
    data_disl_cat = []
    data_prod_sum = []
    data_popular_prod = []

    for party_rk in tqdm(party_list):
        date_series = party2dates[party_rk]
        sum_series = party2sum[party_rk]
        merch_type_series = party2merchant_type[party_rk]
        trans_type_series = party2trans_type[party_rk]

        fin_type_series = party2fin_type[party_rk]
        trans_cat_series = party2trans_cat[party_rk]
        day_series = party2day[party_rk]
        gender_series = party2gender[party_rk]
        age_series = party2age[party_rk]
        marital_status_series = party2marital_status[party_rk]
        child_cnt_series = party2child_cnt[party_rk]
        like_cat_series = party2like_cat[party_rk]
        fav_cat_series = party2fav_cat[party_rk]
        disl_cat_series = party2disl_cat[party_rk]
        prod_sum_series = party2prod_sum[party_rk]
        popular_prod_series = party2popular_prod[party_rk]

        if mode == "train":
            predict_dates = train_predict_dates
        elif mode == "valid":
            predict_dates = valid_predict_dates
        elif mode == "submission":
            predict_dates = submission_predict_dates
        else:
            raise Exception("Unknown mode")

        for predict_period_start in predict_dates:

            predict_period_end = datetime.strftime(
                datetime.strptime(predict_period_start, "%Y-%m-%d")
                + timedelta(days=predict_period_len),
                "%Y-%m-%d",
            )

            l, r = (
                bisect_left(date_series, predict_period_start),
                bisect_right(date_series, predict_period_end),
            )

            history_merch_type = merch_type_series[:l]
            history_sum = sum_series[:l]
            history_trans_type = trans_type_series[:l]
            predict_merch = merch_type_series[l:r]

            history_fin_type = fin_type_series[:l]
            history_trans_cat = trans_cat_series[:l]
            history_day = day_series[:l]
            history_gender = gender_series[:l]
            history_age = age_series[:l]
            history_marital_status = marital_status_series[:l]
            history_child_cnt = child_cnt_series[:l]
            history_like_cat = like_cat_series[:l]
            history_fav_cat = fav_cat_series[:l]
            history_disl_cat = disl_cat_series[:l]
            history_prod_sum = prod_sum_series[:l]
            history_popular_prod = popular_prod_series[:l]

            if predict_merch and l or mode not in ("train", "valid"):
                data_sum.append(history_sum)
                data_trans_type.append(history_trans_type)
                data_merchant_type.append(history_merch_type)
                data_labels.append(predict_merch)

                data_fin_type.append(history_fin_type)
                data_trans_cat.append(history_trans_cat)
                data_day.append(history_day)
                data_gender.append(history_gender)
                data_age.append(history_age)
                data_marital_status.append(history_marital_status)
                data_child_cnt.append(history_child_cnt)
                data_like_cat.append(history_like_cat)
                data_fav_cat.append(history_fav_cat)
                data_disl_cat.append(history_disl_cat)
                data_prod_sum.append(history_prod_sum)
                data_popular_prod.append(history_popular_prod)

    return data_sum, data_trans_type, data_merchant_type, data_labels, \
    data_fin_type, data_trans_cat, data_day, data_gender, data_age, \
    data_marital_status, data_child_cnt, data_like_cat, data_fav_cat, \
    data_disl_cat, data_prod_sum, data_popular_prod

class RSDataset(Dataset):
    def __init__(self, data_sum, data_trans_type, data_merchant_type, labels,
                 data_fin_type, data_trans_cat, data_day, data_gender,
                 data_age, data_marital_status, data_child_cnt, data_like_cat,
                 data_fav_cat, data_disl_cat, data_prod_sum, data_popular_prod):
        super(RSDataset, self).__init__()
        self.data_sum = data_sum
        self.data_trans_type = data_trans_type
        self.data_merchant_type = data_merchant_type
        self.labels = labels

        self.data_fin_type = data_fin_type
        self.data_trans_cat = data_trans_cat
        self.data_day = data_day
        self.data_gender = data_gender
        self.data_age = data_age
        self.data_marital_status = data_marital_status
        self.data_child_cnt = data_child_cnt
        self.data_like_cat = data_like_cat
        self.data_fav_cat = data_fav_cat
        self.data_disl_cat = data_disl_cat
        self.data_prod_sum = data_prod_sum
        self.data_popular_prod = data_popular_prod

    def __len__(self):
        return len(self.data_sum)

    def __getitem__(self, idx):
        targets = np.zeros((MERCH_TYPE_NCLASSES - 1,), dtype=np.float32)
        for m in self.labels[idx]:
            if m:  # skip UNK, UNK-token should not be predicted
                targets[m - 1] = 1.0

        item = {
            "features": {},
            "targets": targets,
        }

        sum_feature = np.array(self.data_sum[idx][-PADDING_LEN:])
        sum_feature = np.vectorize(lambda s: np.log(1 + s))(sum_feature)
        if sum_feature.shape[0] < PADDING_LEN:
            pad = np.zeros(
                (PADDING_LEN - sum_feature.shape[0],), dtype=np.float32
            )
            sum_feature = np.hstack((sum_feature, pad))
        item["features"]["sum"] = torch.from_numpy(sum_feature).float()


        for feature_name, feature_values in zip(
            ["trans_type", "merchant_type", "financial_account_type_cd",
             "transaction_category", "day_of_week", "gender_cd", "age",
             "marital_status_desc", "children_cnt", "most_popular_like_category",
             "most_popular_favorite_category", "most_popular_dislike_category",
             "products_sum", "most_popular_product_chosen"
             ],
            [self.data_trans_type[idx], self.data_merchant_type[idx],
             self.data_fin_type[idx], self.data_trans_cat[idx],
             self.data_day[idx], self.data_gender[idx], self.data_age[idx],
             self.data_marital_status[idx], self.data_child_cnt[idx],
             self.data_like_cat[idx], self.data_fav_cat[idx],
             self.data_disl_cat[idx], self.data_prod_sum[idx],
             self.data_popular_prod[idx]],
        ):

            feature_values = np.array(feature_values[-PADDING_LEN:])
            mask = np.ones(feature_values.shape[0], dtype=np.float32)
            if feature_values.shape[0] < PADDING_LEN:
                feature_values = np.append(
                    feature_values,
                    np.zeros(
                        PADDING_LEN - feature_values.shape[0], dtype=np.int64
                    ),
                )
                mask = np.append(
                    mask,
                    np.zeros(PADDING_LEN - mask.shape[0], dtype=np.float32),
                )
            item["features"][feature_name] = torch.from_numpy(feature_values).long()
            item["features"][f"{feature_name}_mask"] = torch.from_numpy(mask).float()

        return item


if __name__ == '__main__':
    transactions_path = f"{DATADIR}/avk_hackathon_data_transactions.csv"

    DATADIR = "/content/drive/My Drive/datasets/2009-skoltech-hack_data/"
    create_party2 = True
    create_mappings = True

    data_1 = pd.read_csv(transactions_path)

    data = pd.read_csv(DATADIR + 'overall_df_transactions.csv')
    data = data.drop("log_transaction_amt_rur", axis=1)
    data = data.join(data_1["transaction_amt_rur"])
    data.to_csv(f"{DATADIR}/data.csv")
    data_path = f"{DATADIR}/data.csv"

    if create_mappings:
        mappings = defaultdict(dict)
        unk_token = "<UNK>"


        def create_mapping(values):
            mapping = {unk_token: 0}
            for v in values:
                if not pd.isna(v):
                    mapping[str(v)] = len(mapping)

            return mapping


        for col in tqdm(
                [
                    "transaction_type_desc",
                    "merchant_rk",
                    "merchant_type",
                    "merchant_group_rk",
                    "transaction_category",
                    "financial_account_type_cd",
                ]
        ):
            col_values = (
                pd.read_csv(data_path, usecols=[col])[col]
                    .fillna(unk_token)
                    .astype(str)
            )
            mappings[col] = create_mapping(col_values.unique())
            del col_values

        with open(f"{DATADIR}/mappings.json", "w") as f:
            json.dump(mappings, f)

    else:
        with open(f"{DATADIR}/mappings.json", 'r') as f:
            mappings = json.load(f)

    if create_party2:
        # Prepare & save client data
        party2dates = defaultdict(list)  # for each party save a series of the transaction dates
        party2sum = defaultdict(list)  # for each party save a series of the transaction costs
        party2merchant_type = defaultdict(list)  # for each party save a series of the transaction_type
        party2trans_type = defaultdict(list)  # for each party save a series of the transaction merchant_type
        party2fin_type = defaultdict(list)
        party2trans_cat = defaultdict(list)
        party2day = defaultdict(list)
        party2gender = defaultdict(list)
        party2age = defaultdict(list)
        party2marital_status = defaultdict(list)
        party2child_cnt = defaultdict(list)
        party2like_cat = defaultdict(list)
        party2fav_cat = defaultdict(list)
        party2disl_cat = defaultdict(list)
        party2prod_sum = defaultdict(list)
        party2popular_prod = defaultdict(list)

        usecols = [
            "party_rk",
            "transaction_dttm_datetime",
            "transaction_amt_rur",
            "merchant_type",
            "transaction_type_desc",
            "financial_account_type_cd",  # new
            "transaction_category",  # new
            "day_of_week",  # new
            "gender_cd",  # new
            "age",  # new
            "marital_status_desc",  # new
            "children_cnt",  # new
            "most_popular_like_category",  # new
            "most_popular_favorite_category",  # new
            "most_popular_dislike_category",  # new
            "products_sum",  # new
            "most_popular_product_chosen"  # new
        ]

        for chunk in tqdm(
                pd.read_csv(data_path, usecols=usecols, chunksize=100_000)
        ):

            chunk["merchant_type"] = (
                chunk["merchant_type"].fillna(unk_token).astype(str)
            )
            chunk["transaction_type_desc"] = (
                chunk["transaction_type_desc"].fillna(unk_token).astype(str)
            )
            chunk["transaction_amt_rur"] = chunk["transaction_amt_rur"].fillna(0)

            for i, row in chunk.iterrows():
                party2dates[row.party_rk].append(row.transaction_dttm_datetime)
                party2sum[row.party_rk].append(row.transaction_amt_rur)
                party2merchant_type[row.party_rk].append(
                    mappings["merchant_type"][row.merchant_type]
                )
                party2trans_type[row.party_rk].append(
                    mappings["transaction_type_desc"][row.transaction_type_desc]
                )

                party2trans_cat[row.party_rk].append(
                    mappings["transaction_category"][row.transaction_category]
                )

                party2fin_type[row.party_rk].append(row.financial_account_type_cd)
                # party2trans_cat[row.party_rk].append(row.transaction_category)
                party2day[row.party_rk].append(row.day_of_week)
                party2gender[row.party_rk].append(row.gender_cd)
                party2age[row.party_rk].append(row.age)
                party2marital_status[row.party_rk].append(row.marital_status_desc)
                party2child_cnt[row.party_rk].append(row.children_cnt)
                party2like_cat[row.party_rk].append(row.most_popular_like_category)
                party2fav_cat[row.party_rk].append(row.most_popular_favorite_category)
                party2disl_cat[row.party_rk].append(row.most_popular_dislike_category)
                party2prod_sum[row.party_rk].append(row.products_sum)
                party2popular_prod[row.party_rk].append(row.most_popular_product_chosen)

            del chunk

        pickle.dump(party2dates, open(f"{DATADIR}/party2dates.pkl", "wb"))
        pickle.dump(party2sum, open(f"{DATADIR}/party2sum.pkl", "wb"))
        pickle.dump(party2merchant_type, open(f"{DATADIR}/party2merchant_type.pkl", "wb"))
        pickle.dump(party2trans_type, open(f"{DATADIR}/party2trans_type.pkl", "wb"))

        pickle.dump(party2fin_type, open(f"{DATADIR}/party2fin_type.pkl", "wb"))
        pickle.dump(party2trans_cat, open(f"{DATADIR}/party2trans_cat.pkl", "wb"))
        pickle.dump(party2day, open(f"{DATADIR}/party2day.pkl", "wb"))
        pickle.dump(party2gender, open(f"{DATADIR}/party2gender.pkl", "wb"))
        pickle.dump(party2age, open(f"{DATADIR}/party2age.pkl", "wb"))
        pickle.dump(party2marital_status, open(f"{DATADIR}/party2marital_status.pkl", "wb"))
        pickle.dump(party2child_cnt, open(f"{DATADIR}/party2child_cnt.pkl", "wb"))
        pickle.dump(party2like_cat, open(f"{DATADIR}/party2like_cat.pkl", "wb"))
        pickle.dump(party2fav_cat, open(f"{DATADIR}/party2fav_cat.pkl", "wb"))
        pickle.dump(party2disl_cat, open(f"{DATADIR}/party2disl_cat.pkl", "wb"))
        pickle.dump(party2prod_sum, open(f"{DATADIR}/party2prod_sum.pkl", "wb"))
        pickle.dump(party2popular_prod, open(f"{DATADIR}/party2popular_prod.pkl", "wb"))

    else:
        party2dates = pickle.load(open(f"{DATADIR}/party2dates.pkl", 'rb'))
        party2sum = pickle.load(open(f"{DATADIR}/party2sum.pkl", 'rb'))
        party2merchant_type = pickle.load(open(f"{DATADIR}/party2merchant_type.pkl", 'rb'))
        party2trans_type = pickle.load(open(f"{DATADIR}/party2trans_type.pkl", 'rb'))

        party2fin_type = pickle.load(open(f"{DATADIR}/party2fin_type.pkl", 'rb'))
        party2trans_cat = pickle.load(open(f"{DATADIR}/party2trans_cat.pkl", 'rb'))
        party2day = pickle.load(open(f"{DATADIR}/party2day.pkl", 'rb'))
        party2gender = pickle.load(open(f"{DATADIR}/party2gender.pkl", 'rb'))
        party2age = pickle.load(open(f"{DATADIR}/party2age.pkl", 'rb'))
        party2marital_status = pickle.load(open(f"{DATADIR}/party2marital_status.pkl", 'rb'))
        party2child_cnt = pickle.load(open(f"{DATADIR}/party2child_cnt.pkl", 'rb'))
        party2like_cat = pickle.load(open(f"{DATADIR}/party2like_cat.pkl", 'rb'))
        party2fav_cat = pickle.load(open(f"{DATADIR}/party2fav_cat.pkl", 'rb'))
        party2disl_cat = pickle.load(open(f"{DATADIR}/party2disl_cat.pkl", 'rb'))
        party2prod_sum = pickle.load(open(f"{DATADIR}/party2prod_sum.pkl", 'rb'))
        party2popular_prod = pickle.load(open(f"{DATADIR}/party2popular_prod.pkl", 'rb'))

    train_party, valid_party = train_test_split(
        pd.read_csv(transactions_path, usecols=['party_rk']).party_rk.unique(),
        train_size=0.8, random_state=42
    )

    predict_period_len = 60  # -- days
    train_predict_dates = (
        pd.date_range("2019-03-01", "2019-10-31", freq="MS")
            .strftime("%Y-%m-%d")
            .tolist()
    )
    valid_predict_dates = (
        pd.date_range("2019-11-01", "2019-12-31", freq="MS")
            .strftime("%Y-%m-%d")
            .tolist()
    )
    submission_predict_dates = (
        pd.date_range("2020-01-01", "2020-02-28", freq="2MS")
            .strftime("%Y-%m-%d")
            .tolist()
    )

    train_sum, train_trans_type, train_merchant_type, train_labels, train_fin_type, train_trans_cat, train_day, \
    train_gender, train_age, train_marital_status, train_child_cnt, train_like_cat, train_fav_cat, train_disl_cat, \
    train_prod_sum, train_popular_prod = prepare_data(train_party, mode="train")

    valid_sum, valid_trans_type, valid_merchant_type, valid_labels, valid_fin_type, valid_trans_cat, valid_day, \
    valid_gender, valid_age, valid_marital_status, valid_child_cnt, valid_like_cat, valid_fav_cat, valid_disl_cat, \
    valid_prod_sum, valid_popular_prod = prepare_data(valid_party, mode="valid")

    MERCH_TYPE_NCLASSES = len(mappings['merchant_type'])
    TRANS_TYPE_NCLASSES = len(mappings['transaction_type_desc'])
    CAT_TYPE_NCLASSES = len(mappings['transaction_category'])
    PADDING_LEN = 300

    train_dataset = RSDataset(
        train_sum, train_trans_type, train_merchant_type, train_labels,
        train_fin_type, train_trans_cat, train_day, train_gender, train_age,
        train_marital_status, train_child_cnt, train_like_cat, train_fav_cat,
        train_disl_cat, train_prod_sum, train_popular_prod
    )
    valid_dataset = RSDataset(
        valid_sum, valid_trans_type, valid_merchant_type, valid_labels,
        valid_fin_type, valid_trans_cat, valid_day, valid_gender, valid_age,
        valid_marital_status, valid_child_cnt, valid_like_cat, valid_fav_cat,
        valid_disl_cat, valid_prod_sum, valid_popular_prod
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=64, shuffle=False, num_workers=2
    )

