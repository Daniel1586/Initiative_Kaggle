#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
(https://www.kaggle.com/c/ashrae-energy-prediction).
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
    LOCAL_TEST = False
    TARGET = "meter_reading"

    print("========== 2.Load pkl data ...")
    dir_data_pkl = os.getcwd() + "\\predictor-pkl\\"
    train_df = pd.read_pickle(dir_data_pkl + "\\train.pkl")
    infer_df = pd.read_pickle(dir_data_pkl + "\\infer.pkl")
    build_df = pd.read_pickle(dir_data_pkl + "\\build.pkl")
    train_weat_df = pd.read_pickle(dir_data_pkl + "\\weather_train.pkl")
    infer_weat_df = pd.read_pickle(dir_data_pkl + "\\weather_infer.pkl")

    ########################### Building DF merge over concat (to not lose type)
    #################################################################################
    temp_df = train_df[['building_id']]
    temp_df = temp_df.merge(build_df, on=['building_id'], how='left')
    del temp_df['building_id']
    train_df = pd.concat([train_df, temp_df], axis=1)

    temp_df = infer_df[['building_id']]
    temp_df = temp_df.merge(build_df, on=['building_id'], how='left')
    del temp_df['building_id']
    test_df = pd.concat([infer_df, temp_df], axis=1)

    del build_df, temp_df

    ########################### Weather DF merge merge over concat (to not lose type)
    #################################################################################
    temp_df = train_df[['site_id', 'timestamp']]
    temp_df = temp_df.merge(train_weat_df, on=['site_id', 'timestamp'], how='left')
    del temp_df['site_id'], temp_df['timestamp']
    train_df = pd.concat([train_df, temp_df], axis=1)

    temp_df = test_df[['site_id', 'timestamp']]
    temp_df = temp_df.merge(infer_weat_df, on=['site_id', 'timestamp'], how='left')
    del temp_df['site_id'], temp_df['timestamp']
    test_df = pd.concat([test_df, temp_df], axis=1)

    del train_weat_df, infer_weat_df, temp_df

    # Model Features
    rm_cols = ["timestamp",
               TARGET]
    features_cols = [col for col in list(train_df) if col not in rm_cols]
    print(features_cols)

    # Model params
    lgb_params = {
        'objective': 'regression',
        'boosting': 'gbdt',
        'metric': 'rmse',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 8,
        'max_depth': -1,
        'tree_learner': 'serial',
        'colsample_bytree': 0.7,
        'subsample_freq': 1,
        'subsample': 0.7,
        'n_estimators': 600,
        'max_bin': 255,
        'verbose': -1,
        'seed': SEED,
        'early_stopping_rounds': 100,
    }

    # Model Train
    if LOCAL_TEST:
        tr_data = lgb.Dataset(train_df.iloc[:15000000][features_cols],
                              label=np.log1p(train_df.iloc[:15000000][TARGET]))
        vl_data = lgb.Dataset(train_df.iloc[15000000:][features_cols],
                              label=np.log1p(train_df.iloc[15000000:][TARGET]))
        eval_sets = [tr_data, vl_data]
    else:
        tr_data = lgb.Dataset(train_df[features_cols], label=np.log1p(train_df[TARGET]))
        eval_sets = [tr_data]
        estimator = lgb.train(lgb_params, tr_data, valid_sets=eval_sets, verbose_eval=100,)

    if not LOCAL_TEST:
        del tr_data, train_df
        gc.collect()

    if not LOCAL_TEST:
        predictions = []
        batch_size = 2000000
        for batch in range(int(len(test_df) / batch_size) + 1):
            print('Predicting batch:', batch)
            predictions += list(np.expm1(
                estimator.predict(test_df[features_cols].iloc[batch * batch_size:(batch + 1) * batch_size])))

        print('Read submission file and store predictions')
        dir_data_csv = os.getcwd() + "\\great-energy-predictor\\"
        submission = pd.read_csv(dir_data_csv + "\\sample_submission.csv")
        submission['meter_reading'] = predictions
        submission['meter_reading'] = submission['meter_reading'].clip(0, None)
        ########################### Check
        print(submission.iloc[:20])
        print(submission['meter_reading'].describe())
    # Export
    if not LOCAL_TEST:
        submission.to_csv('submission.csv', index=False)
