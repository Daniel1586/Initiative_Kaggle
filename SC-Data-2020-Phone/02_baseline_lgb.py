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
from copy import deepcopy
from sklearn import metrics
from scipy.stats import ks_2samp
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, StratifiedKFold
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
    folds = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED)

    # 数据集划分
    train_x, train_y = tr_df[features_columns], tr_df[target]
    infer_x, infer_y = tt_df[features_columns], tt_df[target]

    # 测试集和验证集处理
    va_df = []
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
        vl_fold = deepcopy(tr_df.iloc[val_idx, :][["phone_no_m", target]])

        estimator = lgb.train(params, tr_data, valid_sets=[tr_data, vl_data], verbose_eval=50)
        valid_p = estimator.predict(vl_x)
        infer_p = estimator.predict(infer_x)
        predictions += infer_p / nfold
        vl_fold["pred"] = valid_p
        va_df.append(vl_fold)
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data, vl_fold
        gc.collect()

    tt_df["label"] = predictions

    return tt_df, va_df


if __name__ == "__main__":
    print("========== 1.Set random seed ...")
    SEED = 32
    set_seed(SEED)

    print("========== 2.Load csv data ...")
    TARGET = "label"
    tra_path = os.getcwd() + "\\data_train.csv"
    tes_path = os.getcwd() + "\\data_tests.csv"
    train_df = pd.read_csv(tra_path, encoding="utf-8")
    infer_df = pd.read_csv(tes_path, encoding="utf-8")

    # Encode Str columns
    for col in list(train_df):
        if train_df[col].dtype == 'O':
            print(col)
            # train_df[col] = train_df[col].fillna("UNK")
            # infer_df[col] = infer_df[col].fillna("UNK")
            # train_df[col] = train_df[col].astype(str)
            # infer_df[col] = infer_df[col].astype(str)
            #
            # le = LabelEncoder()
            # le.fit(list(train_df[col]) + list(infer_df[col]))
            # train_df[col] = le.transform(train_df[col])
            # infer_df[col] = le.transform(infer_df[col])

            train_df[col] = train_df[col].astype("category")
            infer_df[col] = infer_df[col].astype("category")

    # Model Features
    rm_cols = ["phone_no_m", TARGET,
               "county_name", "voc_week_dur_min", "voc_days_max", ]
    features_cols = [col for col in list(train_df) if col not in rm_cols]

    # Model params
    lgb_params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
        'tree_learner': 'serial',
        'seed': SEED,
        'n_estimators': 300,
        'learning_rate': 0.02,
        'max_depth': 5,
        'num_leaves': 24,
        'min_data_in_leaf': 32,
        'bagging_freq': 1,
        'bagging_fraction': 0.75,
        'feature_fraction': 0.7,
        'lambda_l1': 0.01,
        'lambda_l2': 0.01,
        'min_gain_to_split': 5.0,
        'max_bin': 255,
        'verbose': -1,
        'early_stopping_rounds': 100,
    }

    # 单特征筛选
    Feature_Sel = 0
    if Feature_Sel:
        data_list = []
        for i in range(len(features_cols)):
            feat_eva = deepcopy(features_cols)
            del feat_eva[i]
            print(i, len(feat_eva))
            _, valid_pred = make_predictions(train_df, infer_df, feat_eva, TARGET, lgb_params, nfold=5)
            valid_df = pd.concat(valid_pred)
            fpr, tpr, _ = metrics.roc_curve(valid_df[TARGET], valid_df["pred"])
            valid_auc = metrics.auc(fpr, tpr)
            print("\n===== OOF Valid AUC: ", valid_auc)
            valid_df["pred"] = valid_df["pred"].map(lambda x: 1 if x >= 0.3 else 0)
            valid_f1 = metrics.f1_score(valid_df[TARGET], valid_df["pred"], average="macro")
            print("\n===== OOF Valid F1-Score: ", valid_f1)
            data_list.append([i, features_cols[i], valid_auc, valid_f1])
        data_name = ["idx", "feature", "OOF_AUC", "OOF_F1"]
        data_res = pd.DataFrame(columns=data_name, data=data_list)
        data_etl = data_res.sort_values(by=["OOF_AUC", "OOF_F1"], axis=0, ascending=False)
        data_etl.round({"OOF_AUC": 4, "OOF_F1": 4}).to_csv("feat_eva.csv", sep=",", index=False, header=True)

    # 模型训练
    TRAIN_CV = 1
    TRAIN_IF = 1
    if TRAIN_CV:
        print("===== Used feature len:", len(features_cols))
        print("===== Used feature list:", features_cols)
        infer_pred, valid_pred = make_predictions(train_df, infer_df, features_cols, TARGET, lgb_params, nfold=5)
        valid_df = pd.concat(valid_pred)
        fpr, tpr, _ = metrics.roc_curve(valid_df[TARGET], valid_df["pred"])
        valid_auc = metrics.auc(fpr, tpr)
        print("\n===== OOF Valid AUC: ", valid_auc)

        valid_df["pred"] = valid_df["pred"].map(lambda x: 1 if x >= 0.3 else 0)
        valid_f1 = metrics.f1_score(valid_df[TARGET], valid_df["pred"], average="macro")
        print("\n===== OOF Valid F1-Score: ", valid_f1)
        # Export
        if TRAIN_IF:
            infer_pred["label"] = infer_pred["label"].map(lambda x: 1 if x >= 0.3 else 0)
            infer_pred[["phone_no_m", "label"]].to_csv("submit_0706.csv", index=False)

    # 贝叶斯参数优化
    Feature_Opt = 0
    if Feature_Opt:
        opt_lgb = {
            "p1": (100, 400),
            "p2": (0.01, 0.2),
            "p3": (4, 6),
            "p4": (8, 32),
            "p5": (16, 64),
            "p6": (0.5, 0.8),
            "p7": (0.5, 0.8),
            "p8": (0.01, 0.8),
            "p9": (0.01, 0.8),
        }

        def opt_lgb_para(p1, p2, p3, p4, p5, p6, p7, p8, p9):
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            train_x, train_y = train_df[features_cols], train_df[TARGET]
            opt_params = {
                'objective': 'binary',
                'boosting': 'gbdt',
                'metric': 'auc',
                'tree_learner': 'serial',
                'seed': SEED,
                'n_estimators': int(p1),
                'learning_rate': round(p2, 2),
                'max_depth': int(p3),
                'num_leaves': int(p4),
                'min_data_in_leaf': int(p5),
                'bagging_freq': 1,
                'bagging_fraction': round(p6, 2),
                'feature_fraction': round(p7, 2),
                'lambda_l1': round(p8, 2),
                'lambda_l2': round(p9, 2),
                'min_gain_to_split': 5.0,
                'max_bin': 255,
                'verbose': -1,
                'early_stopping_rounds': 100,
            }

            va_df = []
            for fold_, (tra_idx, val_idx) in enumerate(folds.split(train_x, train_y)):
                print("-----Fold:", fold_)
                tr_x, tr_y = train_x.iloc[tra_idx, :], train_y[tra_idx]
                vl_x, vl_y = train_x.iloc[val_idx, :], train_y[val_idx]
                print("-----Train num:", len(tr_x), "Valid num:", len(vl_x))

                tr_data = lgb.Dataset(tr_x, label=tr_y)
                vl_data = lgb.Dataset(vl_x, label=vl_y)
                vl_fold = deepcopy(train_df.iloc[val_idx, :][[TARGET]])

                estimator = lgb.train(opt_params, tr_data, valid_sets=[tr_data, vl_data], verbose_eval=50)
                valid_p = estimator.predict(vl_x)
                vl_fold["pred"] = valid_p
                va_df.append(vl_fold)
                del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data, vl_fold
                gc.collect()

            val_pred = pd.concat(va_df)
            # val_pred["pred"] = val_pred["pred"].map(lambda x: 1 if x >= 0.5 else 0)
            # val_f1 = metrics.f1_score(val_pred[TARGET], val_pred["pred"], average="macro")
            fpr, tpr, _ = metrics.roc_curve(val_pred[TARGET], val_pred["pred"])
            val_auc = metrics.auc(fpr, tpr)

            return val_auc

        rf_bo = BayesianOptimization(f=opt_lgb_para, pbounds=opt_lgb)
        rf_bo.maximize(n_iter=50, init_points=3)
