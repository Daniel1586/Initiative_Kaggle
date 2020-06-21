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

    tol_voc = pd.concat([train_voc, test_voc])
    del train_voc, test_voc, df
    gc.collect()
    phone_no_m = tol_voc[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=["phone_no_m"], keep="first")

    # 每个号码--通话次数/通话的人数(不重复)
    tmp = tol_voc.groupby("phone_no_m")["opposite_no_m"].agg(phone_voc_cnt="count", oppo_voc_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--通话时长的max/min/sum/mean/median/var
    tmp = tol_voc.groupby("phone_no_m")["call_dur"].agg(call_dur_max="max", call_dur_min="min",
                                                        call_dur_sum="sum", call_dur_mean="mean",
                                                        call_dur_median="median", call_dur_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每天[0-31]通话次数的max/min/sum/mean/median/var
    tol_voc["voc_days_cnt"] = tol_voc.groupby(["phone_no_m", "voc_day"])["phone_no_m"].transform("count")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "voc_day"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_days_cnt"].agg(voc_days_max="max", voc_days_min="min",
                                                            voc_days_sum="sum", voc_days_mean="mean",
                                                            voc_days_median="median", voc_days_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每天[0-31]通话时长的max/min/sum/mean/median/var
    tol_voc["voc_days_dur"] = tol_voc.groupby(["phone_no_m", "voc_day"])["call_dur"].transform("sum")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "voc_day"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_days_dur"].agg(voc_days_dur_max="max", voc_days_dur_min="min",
                                                            voc_days_dur_sum="sum", voc_days_dur_mean="mean",
                                                            voc_days_dur_median="median", voc_days_dur_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每小时[0-23]通话次数的max/min/sum/mean/median/var
    tol_voc["voc_hour_cnt"] = tol_voc.groupby(["phone_no_m", "voc_hour"])["phone_no_m"].transform("count")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "voc_hour"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_hour_cnt"].agg(voc_hour_max="max", voc_hour_min="min",
                                                            voc_hour_sum="sum", voc_hour_mean="mean",
                                                            voc_hour_median="median", voc_hour_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每小时[0-23]通话时长的max/min/sum/mean/median/var
    tol_voc["voc_hour_dur"] = tol_voc.groupby(["phone_no_m", "voc_hour"])["call_dur"].transform("sum")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "voc_hour"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_hour_dur"].agg(voc_hour_dur_max="max", voc_hour_dur_min="min",
                                                            voc_hour_dur_sum="sum", voc_hour_dur_mean="mean",
                                                            voc_hour_dur_median="median", voc_hour_dur_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每天[0-6]通话次数的max/min/sum/mean/median/var
    tol_voc["voc_week_cnt"] = tol_voc.groupby(["phone_no_m", "voc_week"])["phone_no_m"].transform("count")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "voc_week"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_week_cnt"].agg(voc_week_max="max", voc_week_min="min",
                                                            voc_week_sum="sum", voc_week_mean="mean",
                                                            voc_week_median="median", voc_week_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每天[0-6]通话时长的max/min/sum/mean/median/var
    tol_voc["voc_week_dur"] = tol_voc.groupby(["phone_no_m", "voc_week"])["call_dur"].transform("sum")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "voc_week"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_week_dur"].agg(voc_week_dur_max="max", voc_week_dur_min="min",
                                                            voc_week_dur_sum="sum", voc_week_dur_mean="mean",
                                                            voc_week_dur_median="median", voc_week_dur_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每个通话对象通话次数的max/min/sum/mean/median/var
    tol_voc["voc_oppo_cnt"] = tol_voc.groupby(["phone_no_m", "opposite_no_m"])["phone_no_m"].transform("count")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "opposite_no_m"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_oppo_cnt"].agg(voc_oppo_max="max", voc_oppo_min="min",
                                                            voc_oppo_sum="sum", voc_oppo_mean="mean",
                                                            voc_oppo_median="median", voc_oppo_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每个通话对象通话时长的max/min/sum/mean/median/var
    tol_voc["voc_oppo_dur"] = tol_voc.groupby(["phone_no_m", "opposite_no_m"])["call_dur"].transform("sum")
    tmp_voc = tol_voc.drop_duplicates(subset=["phone_no_m", "opposite_no_m"], keep="first")
    tmp = tmp_voc.groupby("phone_no_m")["voc_oppo_dur"].agg(voc_oppo_dur_max="max", voc_oppo_dur_min="min",
                                                            voc_oppo_dur_sum="sum", voc_oppo_dur_mean="mean",
                                                            voc_oppo_dur_median="median", voc_oppo_dur_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--通话地市个数(不重复)
    tmp = tol_voc.groupby("phone_no_m")["city_name"].agg(voc_city_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--通话区县个数(不重复)
    tmp = tol_voc.groupby("phone_no_m")["county_name"].agg(voc_county_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--主叫01 通话次数/通话的人数(不重复)/通话次数占比/通话的人数占比/通话地市个数(不重复)/通话区县个数(不重复)
    df_01 = tol_voc[tol_voc["calltype_id"] == 1].copy()
    tmp = df_01.groupby("phone_no_m")["opposite_no_m"].agg(voc_01_cnt="count", voc_01_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    phone_no_m["voc_01_cntR"] = phone_no_m["voc_01_cnt"] / phone_no_m["phone_voc_cnt"]
    phone_no_m["voc_01_unqR"] = phone_no_m["voc_01_unique"] / phone_no_m["oppo_voc_unique"]

    tmp = df_01.groupby("phone_no_m")["city_name"].agg(city_01_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    tmp = df_01.groupby("phone_no_m")["county_name"].agg(county_01_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    return phone_no_m


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

    tol_sms = pd.concat([train_sms, test_sms])
    del train_sms, test_sms, df
    gc.collect()
    phone_no_m = tol_sms[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=["phone_no_m"], keep="first")

    # 每个号码--短信次数/短信的人数(不重复)
    tmp = tol_sms.groupby("phone_no_m")["opposite_no_m"].agg(phone_sms_cnt="count", oppo_sms_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每天[0-31]短信次数的max/min/sum/mean/median/var
    tol_sms["sms_days_cnt"] = tol_sms.groupby(["phone_no_m", "sms_day"])["phone_no_m"].transform("count")
    tmp_sms = tol_sms.drop_duplicates(subset=["phone_no_m", "sms_day"], keep="first")
    tmp = tmp_sms.groupby("phone_no_m")["sms_days_cnt"].agg(sms_days_max="max", sms_days_min="min",
                                                            sms_days_sum="sum", sms_days_mean="mean",
                                                            sms_days_median="median", sms_days_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每小时[0-23]短信次数的max/min/sum/mean/median/var
    tol_sms["sms_hour_cnt"] = tol_sms.groupby(["phone_no_m", "sms_hour"])["phone_no_m"].transform("count")
    tmp_sms = tol_sms.drop_duplicates(subset=["phone_no_m", "sms_hour"], keep="first")
    tmp = tmp_sms.groupby("phone_no_m")["sms_hour_cnt"].agg(sms_hour_max="max", sms_hour_min="min",
                                                            sms_hour_sum="sum", sms_hour_mean="mean",
                                                            sms_hour_median="median", sms_hour_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--每天[0-6]通话次数的max/min/sum/mean/median/var
    tol_sms["sms_week_cnt"] = tol_sms.groupby(["phone_no_m", "sms_week"])["phone_no_m"].transform("count")
    tmp_sms = tol_sms.drop_duplicates(subset=["phone_no_m", "sms_week"], keep="first")
    tmp = tmp_sms.groupby("phone_no_m")["sms_week_cnt"].agg(sms_week_max="max", sms_week_min="min",
                                                            sms_week_sum="sum", sms_week_mean="mean",
                                                            sms_week_median="median", sms_week_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--短信上行01 短信次数/短信的人数(不重复)--相应的占比
    df_01 = tol_sms[tol_sms["calltype_id"] == 1].copy()
    tmp = df_01.groupby("phone_no_m")["opposite_no_m"].agg(sms_01_cnt="count", sms_01_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")
    phone_no_m["sms_01_cntR"] = phone_no_m["sms_01_cnt"] / phone_no_m["phone_sms_cnt"]
    phone_no_m["sms_01_unqR"] = phone_no_m["sms_01_unique"] / phone_no_m["oppo_sms_unique"]

    return phone_no_m


# 上网行为表-----train_app/test_app
def etl_app(path_tr, path_te):
    print("\n========== train_app/test_app ==========\n")
    train_app = pd.read_csv(path_tr + "\\train_app.csv")
    print("----- train_app 大小:", train_app.shape)
    print("----- train_app 列名:", train_app.columns.tolist())

    test_app = pd.read_csv(path_te + "\\test_app.csv")
    print("----- test_app 大小:", test_app.shape)
    print("----- test_app 列名:", test_app.columns.tolist())

    # train_app选取最近一个月数据, 并丢弃多余月份数据
    train_app = train_app[train_app["month_id"] == "2020-03"]
    train_app = train_app.reset_index(drop=True)
    print("----- train_app过滤月份后, 大小:", train_app.shape)

    tol_app = pd.concat([train_app, test_app])
    phone_no_m = tol_app[["phone_no_m"]].copy()
    phone_no_m = phone_no_m.drop_duplicates(subset=["phone_no_m"], keep="first")

    # 每个号码--APP个数/APP个数(不重复)
    tmp = tol_app.groupby("phone_no_m")["busi_name"].agg(phone_app_cnt="count", oppo_app_unique="nunique")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--流量max/min/sum/mean/median/var
    tmp = tol_app.groupby("phone_no_m")["flow"].agg(app_flow_max="max", app_flow_min="min",
                                                    app_flow_sum="sum", app_flow_mean="mean",
                                                    app_flow_median="median", app_flow_var="var")
    phone_no_m = phone_no_m.merge(tmp, on="phone_no_m", how="left")

    # 每个号码--最大流量的APP
    flow = tol_app[["phone_no_m", "busi_name", "flow"]].copy()
    srt_app = flow.sort_values(by=["phone_no_m", "flow"], ascending=False, inplace=False)
    app_flow_max = srt_app.drop_duplicates(subset=["phone_no_m"], keep="first", inplace=False)
    app_flow_max = app_flow_max[["phone_no_m", "busi_name"]]
    app_flow_max.rename(columns={"busi_name": "max_app"}, inplace=True)
    phone_no_m = phone_no_m.merge(app_flow_max, on="phone_no_m", how="left")

    return phone_no_m


if __name__ == "__main__":
    print("\n========== 1.Set random seed ... ==========\n")
    SEED = 42
    set_seed(SEED)

    print("\n========== 2.Load csv data ... ==========\n")
    dir_train = os.getcwd() + "\\train\\"
    dir_tests = os.getcwd() + "\\test\\"

    print("\n========== 3.Merge data ...\n")
    final_user = etl_user(dir_train, dir_tests)
    print("\n----- final_user 大小:", final_user.shape)
    print("----- final_user 列名:", final_user.columns.tolist())
    final_voc = etl_voc(dir_train, dir_tests)
    print("\n----- final_voc 大小:", final_voc.shape)
    print("----- final_voc 列名:", final_voc.columns.tolist())
    final_sms = etl_sms(dir_train, dir_tests)
    print("\n----- final_sms 大小:", final_sms.shape)
    print("----- final_sms 列名:", final_sms.columns.tolist())
    final_app = etl_app(dir_train, dir_tests)
    print("\n----- final_app 大小:", final_app.shape)
    print("----- final_app 列名:", final_app.columns.tolist())

    df_tol = pd.merge(final_user, final_voc, how="left", on="phone_no_m")
    df_tol = pd.merge(df_tol, final_sms, how="left", on="phone_no_m")
    df_tol = pd.merge(df_tol, final_app, how="left", on="phone_no_m")
    print("\n----- df_tol 大小:", df_tol.shape)
    print("----- df_tol 列名:", df_tol.columns.tolist())

    df_train = df_tol[df_tol["label"].notnull()]
    df_tests = df_tol[df_tol["label"].isnull()]
    print("\n----- df_train 大小:", df_train.shape)
    print("----- df_tests 大小:", df_tests.shape)
    df_train.to_csv("data_train.csv", sep=",", index=False, header=True)
    df_tests.to_csv("data_tests.csv", sep=",", index=False, header=True)
