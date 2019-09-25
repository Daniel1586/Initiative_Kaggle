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
from sklearn.model_selection import KFold, GroupKFold
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
    # folds = KFold(n_splits=nfold, shuffle=True, random_state=SEED)
    folds = GroupKFold(n_splits=nfold)

    # 数据集划分
    train_x, train_y = tr_df[features_columns], tr_df[target]
    infer_x, infer_y = tt_df[features_columns], tt_df[target]
    split_groups = tr_df['DT_M']
    tt_df = tt_df[["TransactionID", target]]
    predictions = np.zeros(len(tt_df))
    oof = np.zeros(len(tr_df))

    # 模型训练与预测
    for fold_, (tra_idx, val_idx) in enumerate(folds.split(train_x, train_y, groups=split_groups)):
        print("-----Fold:", fold_)
        tr_x, tr_y = train_x.iloc[tra_idx, :], train_y[tra_idx]
        vl_x, vl_y = train_x.iloc[val_idx, :], train_y[val_idx]
        print("-----Train num:", len(tr_x), "Valid num:", len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        if LOCAL_TEST:
            vl_data = lgb.Dataset(infer_x, label=infer_y)
        else:
            vl_data = lgb.Dataset(vl_x, label=vl_y)
        estimator = lgb.train(params, tr_data, valid_sets=[tr_data, vl_data], verbose_eval=100)
        infer_p = estimator.predict(infer_x)
        predictions += infer_p / nfold
        oof_preds = estimator.predict(vl_x)
        oof[val_idx] = (oof_preds - oof_preds.min()) / (oof_preds.max() - oof_preds.min())

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(), train_x.columns)),
                                       columns=['Value', 'Feature'])
            print(feature_imp)
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()

    tt_df["prediction"] = predictions
    print("OOF AUC:", metrics.roc_auc_score(train_y, oof))

    return tt_df


if __name__ == "__main__":
    print("========== 1.Set random seed ...")
    SEED = 42
    set_seed(SEED)

    print("========== 2.Load pkl data ...")
    LOCAL_TEST = False
    TARGET = "isFraud"
    START_DATE = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")
    dir_data_pkl = os.getcwd() + "\\ieee-fraud-pkl-no-fe\\"
    train_df = pd.read_pickle(dir_data_pkl + "\\train_tran_no_fe.pkl")

    if LOCAL_TEST:
        # Convert TransactionDT to "Month" time-period.
        # We will also drop penultimate block to "simulate" test set values difference
        # TransactionDT时间属性划分,本地测试训练集最后一个月数据为测试集
        train_df["DT_M"] = train_df["TransactionDT"].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        train_df["DT_M"] = (train_df["DT_M"].dt.year - 2017) * 12 + train_df["DT_M"].dt.month
        infer_df = train_df[train_df["DT_M"] == train_df["DT_M"].max()].reset_index(drop=True)
        train_df = train_df[train_df["DT_M"] < (train_df["DT_M"].max() - 1)].reset_index(drop=True)

        train_id_df = pd.read_pickle(dir_data_pkl + "\\train_iden_no_fe.pkl")
        infer_id_df = train_id_df[train_id_df["TransactionID"].isin(infer_df["TransactionID"])].reset_index(drop=True)
        train_id_df = train_id_df[train_id_df["TransactionID"].isin(train_df["TransactionID"])].reset_index(drop=True)
        del train_df["DT_M"], infer_df["DT_M"]
    else:
        infer_df = pd.read_pickle(dir_data_pkl + "\\infer_tran_no_fe.pkl")
        train_id_df = pd.read_pickle(dir_data_pkl + "\\train_iden_no_fe.pkl")
        infer_id_df = pd.read_pickle(dir_data_pkl + "\\infer_iden_no_fe.pkl")

    print("-----Shape control:", train_df.shape, infer_df.shape)
    rm_cols = ["TransactionID", "TransactionDT", TARGET]
    base_columns = [col for col in list(train_df) if col not in rm_cols]

    print("========== 3.Feature Engineering ...")
    ###############################################################################
    # ================================ 增加新特征 ==================================
    # TransactionDT[86400,15811131],START_DATE=2017-11-30
    for df in [train_df, infer_df]:
        df["DT"] = df["TransactionDT"].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
        df["DT_M"] = ((df["DT"].dt.year - 2017) * 12 + df["DT"].dt.month).astype(np.int8)
        df["DT_W"] = ((df["DT"].dt.year - 2017) * 52 + df["DT"].dt.weekofyear).astype(np.int8)
        df["DT_D"] = ((df["DT"].dt.year - 2017) * 365 + df["DT"].dt.dayofyear).astype(np.int16)
        df["DT_day_month"] = df["DT"].dt.day.astype(np.int8)
        df["DT_day_week"] = df["DT"].dt.dayofweek.astype(np.int8)
        df["DT_day_hour"] = df["DT"].dt.hour.astype(np.int8)
        df["Is_december"] = df["DT"].dt.month
        df["Is_december"] = (df["Is_december"] == 12).astype(np.int8)
        df["weekday"] = df["TransactionDT"].map(lambda x: (x // (3600 * 24)) % 7)
    # Total transactions per timeblock
    for col in ["DT_M", "DT_W", "DT_D"]:
        temp_df = pd.concat([train_df[[col]], infer_df[[col]]])
        fq_encode = temp_df[col].value_counts().to_dict()
        train_df[col + "_fq_enc"] = train_df[col].map(fq_encode)
        infer_df[col + "_fq_enc"] = infer_df[col].map(fq_encode)

    # ProductCD[W, C, R, H, S]/card4[visa,..]/card6[debit,..]/M4[M0,M1,M2]
    # 上述特征按类别分组,增加_target_mean特征
    for col in ["ProductCD", "card4", "card6", "M4"]:
        temp_df = train_df.groupby([col])[TARGET].agg(["mean"]).reset_index().rename(
            columns={"mean": col + "_target_mean"})
        temp_df.index = temp_df[col].values
        temp_dict = temp_df[col + "_target_mean"].to_dict()
        train_df[col + "_target_mean"] = train_df[col].map(temp_dict)
        infer_df[col + "_target_mean"] = infer_df[col].map(temp_dict)

    # P_emaildomain/R_emaildomain
    p = "P_emaildomain"
    r = "R_emaildomain"
    ukn = "email_unknown"
    for df in [train_df, infer_df]:
        df[p] = df[p].fillna(ukn)
        df[r] = df[r].fillna(ukn)
        # Check if P_emaildomain matches R_emaildomain
        df["email_check"] = np.where((df[p] == df[r]) & (df[p] != ukn), 1, 0)
        df[p + "_prefix"] = df[p].apply(lambda x: x.split('.')[0])
        df[r + "_prefix"] = df[r].apply(lambda x: x.split('.')[0])

    i_cols = ["ProductCD", "card1", "card2", "card3", "card4", "card5",
              "card6", "addr1", "addr2", "C1", "C2", "C3", "C4", "C5",
              "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14"]
    for col in i_cols:
        temp_df = pd.concat([train_df[[col, "TransactionDT"]], infer_df[[col, "TransactionDT"]]])
        col_count = temp_df.groupby(col)["TransactionDT"].count()
        train_df[col + "_count"] = train_df[col].map(col_count)
        infer_df[col + "_count"] = infer_df[col].map(col_count)

    i_cols = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
    for col in i_cols:
        temp_df = pd.concat([train_df[[col, "TransactionAmt", "C5"]], infer_df[[col, "TransactionAmt", "C5"]]])
        col_count = temp_df.groupby(col)['TransactionAmt'].mean()
        train_df[col + "_amt_count"] = train_df[col].map(col_count)
        infer_df[col + "_amt_count"] = infer_df[col].map(col_count)
        col_count1 = temp_df[temp_df["C5"] == 0].groupby(col)["C5"].count()
        col_count2 = temp_df[temp_df["C5"] != 0].groupby(col)["C5"].count()
        train_df[col + "_C5_count"] = train_df[col].map(col_count2) / (
                    train_df[col].map(col_count1) + 0.01)
        infer_df[col + "_C5_count"] = infer_df[col].map(col_count2) / (
                    infer_df[col].map(col_count1) + 0.01)

    # Let's add some kind of client uID based on cardID ad addr columns
    # The value will be very specific for each client so we need to remove it
    # from final feature. But we can use it for aggregations.
    train_df["uid1"] = train_df["card1"].astype(str) + "_" + train_df["card2"].astype(str)
    infer_df["uid1"] = infer_df["card1"].astype(str) + "_" + infer_df["card2"].astype(str)
    train_df["uid2"] = train_df["uid1"].astype(str) + "_" + train_df["card3"].\
        astype(str) + "_" + train_df["card5"].astype(str)
    infer_df["uid2"] = infer_df["uid1"].astype(str) + "_" + infer_df["card3"].\
        astype(str) + "_" + infer_df["card5"].astype(str)
    train_df["uid3"] = train_df["uid2"].astype(str) + "_" + train_df["addr1"].\
        astype(str) + "_" + train_df["addr2"].astype(str)
    infer_df["uid3"] = infer_df["uid2"].astype(str) + "_" + infer_df["addr1"].\
        astype(str) + "_" + infer_df["addr2"].astype(str)
    train_df["uid4"] = train_df["uid3"].astype(str) + '_' + train_df['P_emaildomain'].astype(str)
    infer_df["uid4"] = infer_df["uid3"].astype(str) + '_' + infer_df['P_emaildomain'].astype(str)
    train_df["uid5"] = train_df["uid3"].astype(str) + '_' + train_df['R_emaildomain'].astype(str)
    infer_df["uid5"] = infer_df["uid3"].astype(str) + '_' + infer_df['R_emaildomain'].astype(str)

    # For our model current TransactionAmt is a noise
    # (even if features importance are telling contrariwise)
    # There are many unique values and model doesn't generalize well
    i_cols = ["card1", "card2", "card3", "card5", "uid1", "uid2", "uid3"]
    for col in i_cols:
        for agg_type in ["mean", "std"]:
            new_col_name = col + "_TransactionAmt_" + agg_type
            temp_df = pd.concat([train_df[[col, "TransactionAmt"]], infer_df[[col, "TransactionAmt"]]])
            temp_df = temp_df.groupby([col])["TransactionAmt"].agg([agg_type]).reset_index().rename(
                columns={agg_type: new_col_name})

            temp_df.index = list(temp_df[col])
            temp_df = temp_df[new_col_name].to_dict()
            train_df[new_col_name] = train_df[col].map(temp_df)
            infer_df[new_col_name] = infer_df[col].map(temp_df)

    # TransactionAmt[0.251, 31937.391]
    train_df["TransactionAmt"] = np.log1p(train_df["TransactionAmt"])
    infer_df["TransactionAmt"] = np.log1p(infer_df["TransactionAmt"])

    ###############################################################################
    # ========================= 合并transaction/identity ==========================
    temp_df1 = train_df[["TransactionID"]]
    temp_df1 = temp_df1.merge(train_id_df, on=["TransactionID"], how="left")
    del temp_df1["TransactionID"]
    train_df = pd.concat([train_df, temp_df1], axis=1)

    temp_df2 = infer_df[["TransactionID"]]
    temp_df2 = temp_df2.merge(infer_id_df, on=["TransactionID"], how="left")
    del temp_df2["TransactionID"]
    infer_df = pd.concat([infer_df, temp_df2], axis=1)

    # Encode Str columns
    for col in list(train_df):
        if train_df[col].dtype == 'O' or infer_df[col].dtype == 'O':
            print("-----category feature", col)
            train_df[col] = train_df[col].fillna("unseen_before_label")
            infer_df[col] = infer_df[col].fillna("unseen_before_label")
            train_df[col] = train_df[col].astype(str)
            infer_df[col] = infer_df[col].astype(str)

            le = LabelEncoder()
            le.fit(list(train_df[col]) + list(infer_df[col]))
            train_df[col] = le.transform(train_df[col])
            infer_df[col] = le.transform(infer_df[col])

            train_df[col] = train_df[col].astype("category")
            infer_df[col] = infer_df[col].astype("category")

    # Final features list
    rm_cols += ["DT", "DT_M", "DT_W", "DT_D", "DT_day_month",
                "uid1", "uid2", "uid3", "uid4", "uid5",
                'V300', 'V309', 'V111', 'V124', 'V106', 'V125',
                'V315', 'V134', 'V102', 'V123', 'V316', 'V113',
                'V136', 'V305', 'V110', 'V299', 'V289', 'V286',
                'V318', 'V304', 'V116', 'V284', 'V293', 'V137',
                'V295', 'V301', 'V104', 'V311', 'V115', 'V109',
                'V119', 'V321', 'V114', 'V133', 'V122', 'V319',
                'V105', 'V112', 'V118', 'V117', 'V121', 'V108',
                'V135', 'V320', 'V303', 'V297', 'V120',   'V1',
                'V14', 'V41', 'V65', 'V88', 'V89', 'V107', 'V68',
                'V28', 'V27', 'V29', 'V241', 'V269', 'V240', 'V325',
                'V138', 'V154', 'V153', 'V330', 'V142', 'V195',
                'V302', 'V328', 'V327', 'V198', 'V196', 'V155']
    features_cols = [col for col in list(train_df) if col not in rm_cols]

    # Model params
    lgb_params = {
        'objective': 'binary',
        'boosting': 'gbdt',
        'metric': 'auc',
        'tree_learner': 'serial',
        'num_threads': 4,
        'seed': SEED,
        'num_iterations': 500,              # 100,number of boosting iterations
        'learning_rate': 0.1,               # 0.1,shrinkage rate
        'num_leaves': 2 ** 9,               # 31,max number of leaves in one tree
        'max_depth': 20,                    # -1,limit the max depth for tree model, -1 means no limit
        'min_data_in_leaf': 20,             # 20,minimal number of data in one leaf
        'min_child_weight': 1e-3,           # 1e-3,minimal sum hessian in one leaf
        'bagging_freq': 1,                  # 0,bagging_fraction/bagging_freq 同时设置才有用
        'bagging_fraction': 1.0,            # 1.0,randomly select part of data without resampling
        'feature_fraction': 0.7,            # 1.0,randomly select part of features on each iteration
        'lambda_l1': 0.1,                   # 0.0,L1 regularization
        'lambda_l2': 1.0,                   # 0.0,L2 regularization
        'min_gain_to_split': 0.0,           # 0.0,the minimal gain to perform split
        'cat_smooth': 10.0,                 # 10.0,used for the categorical features
        'early_stopping_round': 100,
        'max_bin': 255,
        'verbose': -1,
    }

    # Model Train
    if LOCAL_TEST:
        test_predictions = make_predictions(train_df, infer_df, features_cols, TARGET, lgb_params)
        print(metrics.roc_auc_score(test_predictions[TARGET], test_predictions["prediction"]))
    else:
        print("-----Shape control:", train_df.shape, infer_df.shape)
        print("-----Used features:", len(features_cols))
        lgb_params["max_depth"] = 16

        test_predictions = make_predictions(train_df, infer_df, features_cols, TARGET, lgb_params, nfold=6)
    # Export
    if not LOCAL_TEST:
        test_predictions["isFraud"] = test_predictions["prediction"]
        test_predictions[["TransactionID", "isFraud"]].to_csv("092502.csv", index=False)
