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
from scipy.stats import ks_2samp
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
    tt_df = tt_df[["phone_no_m"]]
    predictions = np.zeros(len(tt_df))

    # 模型训练与预测
    for fold_, (tra_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
        print("-----Fold:", fold_)
        tr_x, tr_y = train_x.iloc[tra_idx, :], train_y[tra_idx]
        vl_x, vl_y = train_x.iloc[val_idx, :], train_y[val_idx]
        print("-----Train num:", len(tr_x), "Valid num:", len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)

        estimator = lgb.train(params, tr_data, valid_sets=[tr_data, vl_data], verbose_eval=200)
        infer_p = estimator.predict(infer_x)
        predictions += infer_p / nfold
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()

    tt_df["label"] = predictions

    return tt_df


if __name__ == "__main__":
    print("========== 1.Set random seed ...")
    SEED = 42
    set_seed(SEED)

    print("========== 2.Load csv data ...")
    TARGET = "label"
    tra_path = os.getcwd() + "\\data_train.csv"
    tes_path = os.getcwd() + "\\data_tests.csv"
    train_df = pd.read_csv(tra_path, encoding="utf-8")
    infer_df = pd.read_csv(tes_path, encoding="utf-8")

    # Encode Str columns
    # for col in list(train_df):
    #     if train_df[col].dtype == 'O':
    #         print(col)
    #         train_df[col] = train_df[col].fillna("unseen_before_label")
    #         infer_df[col] = infer_df[col].fillna("unseen_before_label")
    #         train_df[col] = train_df[col].astype(str)
    #         infer_df[col] = infer_df[col].astype(str)
    #
    #         le = LabelEncoder()
    #         le.fit(list(train_df[col]) + list(infer_df[col]))
    #         train_df[col] = le.transform(train_df[col])
    #         infer_df[col] = le.transform(infer_df[col])
    #
    #         train_df[col] = train_df[col].astype("category")
    #         infer_df[col] = infer_df[col].astype("category")

    # Model Features
    rm_cols = ["phone_no_m", TARGET]
    features_cols = [col for col in list(train_df) if col not in rm_cols]

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
        'subsample': 0.7,
        'n_estimators': 800,
        'max_bin': 255,
        'verbose': -1,
        'seed': SEED,
        'early_stopping_rounds': 100,
    }

    # Model Train
    TRAIN_CV = 1
    TRAIN_IF = 1
    if TRAIN_CV:
        print("-----Shape control:", train_df.shape, infer_df.shape)
        lgb_params["learning_rate"] = 0.01
        lgb_params["n_estimators"] = 1000
        lgb_params["early_stopping_rounds"] = 100
        test_predictions = make_predictions(train_df, infer_df, features_cols, TARGET, lgb_params, nfold=5)
    # Export
    if TRAIN_IF:
        test_predictions[["phone_no_m", "label"]].to_csv("submit_0612.csv", index=False)
