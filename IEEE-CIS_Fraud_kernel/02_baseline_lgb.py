#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Preprocess ieee-fraud-detection dataset.
(https://www.kaggle.com/c/ieee-fraud-detection).
Train shape:(590540,394),identity(144233,41)--isFraud 3.5%
Test  shape:(506691,393),identity(141907,41)
############### TF Version: 1.13.1/Python Version: 3.7 ###############
"""

import os
import gc
import random
import warnings
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


# make all processes deterministic/固定随机数生成器的种子
# environ是一个字符串所对应环境的映像对象,PYTHONHASHSEED为其中的环境变量
# Python会用一个随机的种子来生成str/bytes/datetime对象的hash值;
# 如果该环境变量被设定为一个数字,它就被当作一个固定的种子来生成str/bytes/datetime对象的hash值
def set_seed(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def make_predictions(tr_df, tt_df, features_columns, target, params, nfold=2):
    # K折交叉验证
    folds = KFold(n_splits=nfold, shuffle=True, random_state=SEED)

    # 数据集划分
    train_x, train_y = tr_df[features_columns], tr_df[target]
    infer_x, infer_y = tt_df[features_columns], tt_df[target]
    tt_df = tt_df[["TransactionID", target]]
    predictions = np.zeros(len(tt_df))

    # 模型训练与预测
    for fold_, (tra_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
        print("-----Fold:", fold_)
        tr_x, tr_y = train_x.iloc[tra_idx, :], train_y[tra_idx]
        vl_x, vl_y = train_x.iloc[val_idx, :], train_y[val_idx]
        print("-----Train num:", len(tr_x), "Valid num:", len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(infer_x, label=infer_y)
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)
        estimator = lgb.train(params, tr_data, valid_sets=[tr_data, vl_data], verbose_eval=200)
        infer_p = estimator.predict(infer_x)
        predictions += infer_p / nfold

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), train_x.columns)),
                                       columns=['Value', 'Feature'])
            print(feature_imp)
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()

    tt_df["prediction"] = predictions

    return tt_df


if __name__ == "__main__":
    print("========== 1.Set random seed ...")
    SEED = 42
    set_seed(SEED)

    print("========== 2.Load pkl data ...")
    LOCAL_TEST = False
    TARGET = "isFraud"
    START_DATE = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")
    dir_data_pkl = os.getcwd() + "\\ieee-fraud-pkl\\"
    train_df = pd.read_pickle(dir_data_pkl + "\\train_transaction.pkl")

    if LOCAL_TEST:
        # Convert TransactionDT to "Month" time-period.
        # We will also drop penultimate block to "simulate" test set values difference
        # TransactionDT时间属性划分,本地测试训练集最后一个月数据为测试集
        train_df["DT_M"] = train_df["TransactionDT"].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        train_df["DT_M"] = (train_df["DT_M"].dt.year - 2017) * 12 + train_df["DT_M"].dt.month
        infer_df = train_df[train_df["DT_M"] == train_df["DT_M"].max()].reset_index(drop=True)
        train_df = train_df[train_df["DT_M"] < (train_df["DT_M"].max() - 1)].reset_index(drop=True)

        train_id_df = pd.read_pickle(dir_data_pkl + "\\train_identity.pkl")
        infer_id_df = train_id_df[train_id_df["TransactionID"].isin(infer_df["TransactionID"])].reset_index(drop=True)
        train_id_df = train_id_df[train_id_df["TransactionID"].isin(train_df["TransactionID"])].reset_index(drop=True)
        del train_df["DT_M"], infer_df["DT_M"]
    else:
        infer_df = pd.read_pickle(dir_data_pkl + "\\infer_transaction.pkl")
        train_id_df = pd.read_pickle(dir_data_pkl + "\\train_identity.pkl")
        infer_id_df = pd.read_pickle(dir_data_pkl + "\\infer_identity.pkl")

    # TransactionDT and D9
    # Also, seems that D9 column is an hour and it is the same as df['DT'].dt.hour
    for df in [train_df, infer_df]:
        df["DT"] = df["TransactionDT"].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        df["DT_M"] = (df["DT"].dt.year - 2017) * 12 + df["DT"].dt.month
        df["DT_W"] = (df["DT"].dt.year - 2017) * 52 + df["DT"].dt.weekofyear
        df["DT_D"] = (df["DT"].dt.year - 2017) * 365 + df["DT"].dt.dayofyear

        df["DT_hour"] = df["DT"].dt.hour
        df["DT_day_week"] = df["DT"].dt.dayofweek
        df["DT_day"] = df["DT"].dt.day

        # D9 column
        df["D9"] = np.where(df["D9"].isna(), 0, 1)

    # Reset values for "noise" card1
    i_cols = ["card1"]
    for col in i_cols:
        valid_card = pd.concat([train_df[[col]], infer_df[[col]]])
        valid_card = valid_card[col].value_counts()
        valid_card = valid_card[valid_card > 2]
        valid_card = list(valid_card.index)

        train_df[col] = np.where(train_df[col].isin(infer_df[col]), train_df[col], np.nan)
        infer_df[col] = np.where(infer_df[col].isin(train_df[col]), infer_df[col], np.nan)
        train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
        infer_df[col] = np.where(infer_df[col].isin(valid_card), infer_df[col], np.nan)

    # M columns (except M4)
    # All these columns are binary encoded 1/0
    i_cols = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]
    for df in [train_df, infer_df]:
        df["M_sum"] = df[i_cols].sum(axis=1).astype(np.int8)
        df["M_nan"] = df[i_cols].isna().sum(axis=1).astype(np.int8)

    # Freq encoding
    i_cols = ['card1', 'card2', 'card3', 'card5',
              'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',
              'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9',
              'addr1', 'addr2',
              'dist1', 'dist2',
              'P_emaildomain', 'R_emaildomain']

    for col in i_cols:
        temp_df = pd.concat([train_df[[col]], infer_df[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
        train_df[col + '_fq_enc'] = train_df[col].map(fq_encode)
        infer_df[col + '_fq_enc'] = infer_df[col].map(fq_encode)

    # ProductCD and M4 Target mean
    for col in ['ProductCD', 'M4']:
        temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
            columns={'mean': col + '_target_mean'})
        temp_dict.index = temp_dict[col].values
        temp_dict = temp_dict[col + '_target_mean'].to_dict()
        train_df[col + '_target_mean'] = train_df[col].map(temp_dict)
        infer_df[col + '_target_mean'] = infer_df[col].map(temp_dict)

    # Encode Str columns
    for col in list(train_df):
        if train_df[col].dtype == 'O':
            print(col)
            train_df[col] = train_df[col].fillna('unseen_before_label')
            infer_df[col] = infer_df[col].fillna('unseen_before_label')

            train_df[col] = train_df[col].astype(str)
            infer_df[col] = infer_df[col].astype(str)

            le = LabelEncoder()
            le.fit(list(train_df[col]) + list(infer_df[col]))
            train_df[col] = le.transform(train_df[col])
            infer_df[col] = le.transform(infer_df[col])

            train_df[col] = train_df[col].astype('category')
            infer_df[col] = infer_df[col].astype('category')

    # Model Features
    # We can use set().difference() but order matters
    rm_cols = ['TransactionID', 'TransactionDT', TARGET]
    features_cols = list(train_df)
    for col in rm_cols:
        if col in features_cols:
            features_cols.remove(col)

    # Model params
    lgb_params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.01,
        'num_leaves': 2 ** 8,
        'max_depth': -1,
        'tree_learner': 'serial',
        'colsample_bytree': 0.7,
        'subsample_freq': 1,
        'subsample': 1,
        'n_estimators': 800,
        'max_bin': 255,
        'verbose': -1,
        'seed': SEED,
        'early_stopping_rounds': 100,
    }

    # Model Train
    if LOCAL_TEST:
        test_predictions = make_predictions(train_df, infer_df, features_cols, TARGET, lgb_params)
        print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions["prediction"]))
    else:
        lgb_params["learning_rate"] = 0.01
        lgb_params["n_estimators"] = 800
        lgb_params["early_stopping_rounds"] = 100
        test_predictions = make_predictions(train_df, infer_df, features_cols, TARGET, lgb_params, nfold=5)
    # Export
    if not LOCAL_TEST:
        test_predictions["isFraud"] = test_predictions["prediction"]
        test_predictions[["TransactionID", "isFraud"]].to_csv("091303.csv", index=False)
