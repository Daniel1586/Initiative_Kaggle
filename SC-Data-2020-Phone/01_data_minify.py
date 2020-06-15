#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
############### TF Version: 1.13.1/Python Version: 3.7 ###############
"""

import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
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


# [train+infer]离散特征编码:[NaN不编码]
def minify_identity_df(df):
    df['id_12'] = df['id_12'].map({'Found': 1, 'NotFound': 0})
    df['id_15'] = df['id_15'].map({'New': 2, 'Found': 1, 'Unknown': 0})
    df['id_16'] = df['id_16'].map({'Found': 1, 'NotFound': 0})
    df['id_23'] = df['id_23'].map({'IP_PROXY:TRANSPARENT': 3, 'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 1})

    df['id_27'] = df['id_27'].map({'Found': 1, 'NotFound': 0})
    df['id_28'] = df['id_28'].map({'New': 2, 'Found': 1})
    df['id_29'] = df['id_29'].map({'Found': 1, 'NotFound': 0})

    df['id_35'] = df['id_35'].map({'T': 1, 'F': 0})
    df['id_36'] = df['id_36'].map({'T': 1, 'F': 0})
    df['id_37'] = df['id_37'].map({'T': 1, 'F': 0})
    df['id_38'] = df['id_38'].map({'T': 1, 'F': 0})

    df['id_34'] = df['id_34'].fillna(':3')
    df['id_34'] = df['id_34'].apply(lambda x: x.split(':')[1]).astype(np.int8)
    df['id_34'] = np.where(df['id_34'] == 3, np.nan, df['id_34'])

    df['id_33'] = df['id_33'].fillna('0x0')
    df['id_33_0'] = df['id_33'].apply(lambda x: x.split('x')[0]).astype(int)
    df['id_33_1'] = df['id_33'].apply(lambda x: x.split('x')[1]).astype(int)
    df['id_33'] = np.where(df['id_33'] == '0x0', np.nan, df['id_33'])

    df['DeviceType'].map({'desktop': 1, 'mobile': 0})
    return df


# 基础资料表-----train_user/test_user
def etl_user(path_tr, path_te):
    print("\n========== train_user/test_user ==========\n")
    train_user = pd.read_csv(path_tr + "\\train_user.csv")
    train_user__ = train_user.drop_duplicates(subset=["phone_no_m"], keep="first", inplace=False)
    print("----- train_user 大小:", train_user.shape)
    print("----- train_user 列名:", train_user.columns.tolist())
    print("----- train_user 字段 phone_no_m 无重复!", train_user__.shape)
    print("----- train_user 0/1标签数量\n", train_user.label.value_counts())
    tr_user = set(list(train_user__.phone_no_m))

    test_user = pd.read_csv(path_te + "\\test_user.csv")
    test_user__ = test_user.drop_duplicates(subset=["phone_no_m"], keep="first", inplace=False)
    print("----- test_user 大小:", test_user.shape)
    print("----- test_user 列名:", test_user.columns.tolist())
    print("----- test_user 字段 phone_no_m 无重复!", test_user__.shape)
    te_user = set(list(test_user__.phone_no_m))
    print("----- train_user和test_user 字段phone_no_m相交的数量:", len(tr_user & te_user))

    del train_user__, tr_user, test_user__, te_user
    gc.collect()

    # train_user选取最近一个月数据, 并丢弃多余月份数据
    train_user["arpu_202004"] = train_user["arpu_202003"]
    train_user.drop(["arpu_201908", "arpu_201909", "arpu_201910", "arpu_201911",
                     "arpu_201912", "arpu_202001", "arpu_202002", "arpu_202003"], axis=1, inplace=True)

    tol_user = pd.concat([train_user, test_user])
    # # labelEncoder标准化标签
    # for col1 in ["city_name", "county_name"]:
    #     tol_user[col1] = tol_user[col1].fillna("UNK")
    #     le = LabelEncoder()
    #     le.fit(list(tol_user[col1]))
    #     tol_user[col1] = le.transform(tol_user[col1])
    print("----- tol_user 大小:", tol_user.shape)
    print("----- tol_user 列名:", tol_user.columns.tolist())

    return tol_user


# 语音通话表-----train_voc/test_voc
def etl_voc(path_tr, path_te):
    print("\n========== train_voc/test_voc ==========\n")
    train_voc = pd.read_csv(path_tr + "\\train_voc.csv")
    train_voc["start_datetime"] = train_voc["start_datetime"].astype("datetime64")
    print("----- train_voc 大小:", train_voc.shape)
    print("----- train_voc 列名:", train_voc.columns.tolist())

    test_voc = pd.read_csv(path_te + "\\test_voc.csv")
    test_voc["start_datetime"] = test_voc["start_datetime"].astype("datetime64")
    print("----- test_voc 大小:", test_voc.shape)
    print("----- test_voc 列名:", test_voc.columns.tolist())

    # train_voc选取最近一个月数据, 并丢弃多余月份数据
    train_voc = train_voc[train_voc["start_datetime"] >= "2020-03-01 00:00:00"]
    train_voc = train_voc.reset_index(drop=True)
    print("----- train_voc过滤月份后, 大小:", train_voc.shape)

    # start_datetime 通话日期
    for df in [train_voc, test_voc]:
        df["voc_day"] = df["start_datetime"].dt.day
        df["voc_hour"] = df["start_datetime"].dt.hour
        df["voc_week"] = df["start_datetime"].dt.dayofweek

    # 按号码/天/小时/周/对端号码/通话类型--统计通话次数
    tol_voc = pd.concat([train_voc, test_voc])
    tol_voc["voc_phone_cnt"] = tol_voc.groupby(["phone_no_m"])["phone_no_m"].transform("count")
    tol_voc["voc_day_cnt"] = tol_voc.groupby(["phone_no_m", "voc_day"])["phone_no_m"].transform("count")
    tol_voc["voc_hour_cnt"] = tol_voc.groupby(["phone_no_m", "voc_hour"])["phone_no_m"].transform("count")
    tol_voc["voc_week_cnt"] = tol_voc.groupby(["phone_no_m", "voc_week"])["phone_no_m"].transform("count")
    tol_voc["voc_oppo_cnt"] = tol_voc.groupby(["phone_no_m", "opposite_no_m"])["phone_no_m"].transform("count")
    tol_voc["voc_type_cnt"] = tol_voc.groupby(["phone_no_m", "calltype_id"])["phone_no_m"].transform("count")
    tol_voc["voc_city_cnt"] = tol_voc.groupby(["phone_no_m", "city_name"])["phone_no_m"].transform("count")
    tol_voc["voc_county_cnt"] = tol_voc.groupby(["phone_no_m", "city_name"])["phone_no_m"].transform("count")
    del train_voc, test_voc, df
    gc.collect()

    i_cols = ["voc_day_cnt", "voc_hour_cnt", "voc_week_cnt", "voc_oppo_cnt",
              "voc_type_cnt", "call_dur", "voc_city_cnt", "voc_county_cnt"]
    for col in i_cols:
        for agg_type in ["mean", "std", "max", "min"]:
            new_col_name = col + "_" + agg_type
            tol_voc[new_col_name] = tol_voc.groupby(["phone_no_m"])[col].transform(agg_type)
    print("----- tol_voc 大小:", tol_voc.shape)
    print("----- tol_voc 列名:", tol_voc.columns.tolist())
    return tol_voc


# 短信表-----train_sms/test_sms
def etl_sms(path_tr, path_te):
    print("\n========== train_sms/test_sms ==========\n")
    train_sms = pd.read_csv(path_tr + "\\train_sms.csv")
    train_sms["request_datetime"] = train_sms["request_datetime"].astype("datetime64")
    print("----- train_sms 大小:", train_sms.shape)
    print("----- train_sms 列名:", train_sms.columns.tolist())

    test_sms = pd.read_csv(path_te + "\\test_sms.csv")
    test_sms["request_datetime"] = test_sms["request_datetime"].astype("datetime64")
    print("----- test_sms 大小:", test_sms.shape)
    print("----- test_sms 列名:", test_sms.columns.tolist())

    # train_sms选取最近一个月数据, 并丢弃多余月份数据
    train_sms = train_sms[train_sms["request_datetime"] >= "2020-03-01 00:00:00"]
    train_sms = train_sms.reset_index(drop=True)
    print("----- train_sms过滤月份后, 大小:", train_sms.shape)

    # request_datetime 短信发送日期
    for df in [train_sms, test_sms]:
        df["sms_day"] = df["request_datetime"].dt.day
        df["sms_hour"] = df["request_datetime"].dt.hour
        df["sms_week"] = df["request_datetime"].dt.dayofweek

    # 按号码/天/小时/周统计通话次数
    tol_sms = pd.concat([train_sms, test_sms])
    tol_sms["sms_phone_cnt"] = tol_sms.groupby(["phone_no_m"])["phone_no_m"].transform("count")
    tol_sms["sms_day_cnt"] = tol_sms.groupby(["phone_no_m", "sms_day"])["phone_no_m"].transform("count")
    tol_sms["sms_hour_cnt"] = tol_sms.groupby(["phone_no_m", "sms_hour"])["phone_no_m"].transform("count")
    tol_sms["sms_week_cnt"] = tol_sms.groupby(["phone_no_m", "sms_week"])["phone_no_m"].transform("count")
    del train_sms, test_sms, df
    gc.collect()

    i_cols = ["sms_day_cnt", "sms_hour_cnt", "sms_week_cnt"]
    for col in i_cols:
        for agg_type in ["mean", "std", "max", "min"]:
            new_col_name = col + "_" + agg_type
            tol_sms[new_col_name] = tol_sms.groupby(["phone_no_m"])[col].transform(agg_type)
    print("----- tol_sms 大小:", tol_sms.shape)
    print("----- tol_sms 列名:", tol_sms.columns.tolist())
    return tol_sms


# 上网行为表-----train_app/test_app
def etl_app(path_tr, path_te):
    print("\n========== train_app/test_app ==========\n")
    train_app = pd.read_csv(dir_train + "\\train_app.csv")
    print("----- train_app 大小:", train_app.shape)
    print("----- train_app 列名:", train_app.columns.tolist())

    test_app = pd.read_csv(dir_tests + "\\test_app.csv")
    print("----- test_app 大小:", test_app.shape)
    print("----- test_app 列名:", test_app.columns.tolist())

    # train_app选取最近一个月数据, 并丢弃多余月份数据
    train_app = train_app[train_app["month_id"] == "2020-03"]
    train_app = train_app.reset_index(drop=True)
    print("----- train_app过滤月份后, 大小:", train_app.shape)

    # 按号码统计流量
    tol_app = pd.concat([train_app, test_app])
    tol_app["app_phone_cnt"] = tol_app.groupby(["phone_no_m"])["phone_no_m"].transform("count")
    i_cols = ["flow"]
    for col in i_cols:
        for agg_type in ["mean", "std", "max", "min", "sum"]:
            new_col_name = col + "_" + agg_type
            tol_app[new_col_name] = tol_app.groupby(["phone_no_m"])[col].transform(agg_type)
    print("----- tol_app 大小:", tol_app.shape)
    print("----- tol_app 列名:", tol_app.columns.tolist())

    return tol_app


if __name__ == "__main__":
    print("\n========== 1.Set random seed ... ==========\n")
    SEED = 42
    set_seed(SEED)

    print("\n========== 2.Load csv data ... ==========\n")
    dir_train = os.getcwd() + "\\train\\"
    dir_tests = os.getcwd() + "\\test\\"

    # vld_user = etl_user(dir_train, dir_tests)
    vld_voc = etl_voc(dir_train, dir_tests)
    # vld_sms = etl_sms(dir_train, dir_tests)
    # vld_app = etl_app(dir_train, dir_tests)

    # print("\n========== 3.Merge data ...\n")
    # # 取有效特征
    # tol_voc = tol_voc[["phone_no_m", "calltype_id", "call_dur", "voc_day", "voc_hour", "voc_week",
    #                    "voc_phone_cnt", "voc_day_cnt", "voc_hour_cnt", "voc_week_cnt",
    #                    "voc_day_cnt_mean", "voc_day_cnt_std", "voc_day_cnt_max", "voc_day_cnt_min",
    #                    "voc_hour_cnt_mean", "voc_hour_cnt_std", "voc_hour_cnt_max", "voc_hour_cnt_min",
    #                    "voc_week_cnt_mean", "voc_week_cnt_std", "voc_week_cnt_max", "voc_week_cnt_min"]]
    # vld_voc = tol_voc.drop_duplicates(["phone_no_m"], keep="first", inplace=False)
    # print("----- vld_voc 大小:", vld_voc.shape)
    # print("----- vld_voc 列名:", vld_voc.columns.tolist())
    #
    # tol_sms = tol_sms[["phone_no_m", "calltype_id", "sms_day", "sms_hour", "sms_week", "sms_phone_cnt",
    #                    "sms_day_cnt", "sms_hour_cnt", "sms_week_cnt",
    #                    "sms_day_cnt_mean", "sms_day_cnt_std", "sms_day_cnt_max", "sms_day_cnt_min",
    #                    "sms_hour_cnt_mean", "sms_hour_cnt_std", "sms_hour_cnt_max", "sms_hour_cnt_min",
    #                    "sms_week_cnt_mean", "sms_week_cnt_std", "sms_week_cnt_max", "sms_week_cnt_min"]]
    # vld_sms = tol_sms.drop_duplicates(["phone_no_m"], keep="first", inplace=False)
    # print("----- vld_sms 大小:", vld_sms.shape)
    # print("----- vld_sms 列名:", vld_sms.columns.tolist())
    #
    # tol_app = tol_app[["phone_no_m", "app_phone_cnt", "flow_mean", "flow_std", "flow_max",
    #                    "flow_min", "flow_sum"]]
    # vld_app = tol_app.drop_duplicates(["phone_no_m"], keep="first", inplace=False)
    # print("----- vld_app 大小:", vld_app.shape)
    # print("----- vld_app 列名:", vld_app.columns.tolist())
    # df_tol = pd.merge(tol_user, vld_voc, how="left", on="phone_no_m")
    # df_tol = pd.merge(df_tol, vld_sms, how="left", on="phone_no_m")
    # df_tol = pd.merge(df_tol, vld_app, how="left", on="phone_no_m")
    # print("----- df_tol 大小:", df_tol.shape)
    # print("----- df_tol 列名:", df_tol.columns.tolist())
    # df_train = df_tol[df_tol["label"].notnull()]
    # df_tests = df_tol[df_tol["label"].isnull()]
    # print("----- df_train 大小:", df_train.shape)
    # print("----- df_tests 大小:", df_tests.shape)
    # df_train.to_csv("data_train.csv", sep=",", index=False, header=True)
    # df_tests.to_csv("data_tests.csv", sep=",", index=False, header=True)
