#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
(https://www.kaggle.com/c/ashrae-energy-prediction).
Train shape:(590540,394),identity(144233,41)--isFraud 3.5%
Test  shape:(506691,393),identity(141907,41)
############### TF Version: 1.13.1/Python Version: 3.7 ###############
"""

import os
import math
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


# reduce memory for dataframe/优化dataframe数据格式,减少内存占用
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100*(start_mem-end_mem)/start_mem
    if verbose:
        print("Default Mem. {:.2f} Mb, Optimized Mem. {:.2f} Mb, Reduction {:.1f}%".
              format(start_mem, end_mem, reduction))
    return df


if __name__ == "__main__":
    print("========== 1.Set random seed ...")
    SEED = 42
    set_seed(SEED)
    LOCAL_TEST = False

    print("========== 2.Load csv data ...")
    dir_data_csv = os.getcwd() + "\\great-energy-predictor\\"
    train_df = pd.read_csv(dir_data_csv + "\\train.csv")
    infer_df = pd.read_csv(dir_data_csv + "\\test.csv")
    build_df = pd.read_csv(dir_data_csv + "\\building_metadata.csv")
    train_weat_df = pd.read_csv(dir_data_csv + "\\weather_train.csv")
    infer_weat_df = pd.read_csv(dir_data_csv + "\\weather_test.csv")

    print('#' * 30)
    print('Main data:', list(train_df), train_df.info())
    print('#' * 30)
    print('Buildings data:', list(build_df), build_df.info())
    print('#' * 30)
    print('Weather data:', list(train_weat_df), train_weat_df.info())
    print('#' * 30)

    for df in [train_df, infer_df, train_weat_df, infer_weat_df]:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    for df in [train_df, infer_df]:
        df["DT_M"] = df["timestamp"].dt.month.astype(np.int8)
        df["DT_W"] = df["timestamp"].dt.weekofyear.astype(np.int8)
        df["DT_D"] = df["timestamp"].dt.dayofyear.astype(np.int16)

        df["DT_hour"] = df["timestamp"].dt.hour.astype(np.int8)
        df["DT_day_week"] = df["timestamp"].dt.dayofweek.astype(np.int8)
        df["DT_day_month"] = df["timestamp"].dt.day.astype(np.int8)
        df["DT_week_month"] = df["timestamp"].dt.day / 7
        df["DT_week_month"] = df["DT_week_month"].apply(lambda x: math.ceil(x)).astype(np.int8)

    print("========== 3.ETL tran categorical feature [String] ...")
    build_df['primary_use'] = build_df['primary_use'].astype('category')
    build_df['floor_count'] = build_df['floor_count'].fillna(0).astype(np.int8)
    build_df['year_built'] = build_df['year_built'].fillna(-999).astype(np.int16)

    le = LabelEncoder()
    build_df['primary_use'] = build_df['primary_use'].astype(str)
    build_df['primary_use'] = le.fit_transform(build_df['primary_use']).astype(np.int8)
    do_not_convert = ['category', 'datetime64[ns]', 'object']
    for df in [train_df, infer_df, build_df, train_weat_df, infer_weat_df]:
        original = df.copy()
        df = reduce_mem_usage(df)
        for col in list(df):
            if df[col].dtype.name not in do_not_convert:
                if (df[col] - original[col]).sum() != 0:
                    df[col] = original[col]
                    print('Bad transformation', col)

    print('#' * 30)
    print('Main data:', list(train_df), train_df.info())
    print('#' * 30)
    print('Buildings data:', list(build_df), build_df.info())
    print('#' * 30)
    print('Weather data:', list(train_weat_df), train_weat_df.info())
    print('#' * 30)

    print("========== 4.Save pkl ...")
    train_df.to_pickle("train.pkl")
    infer_df.to_pickle("infer.pkl")
    build_df.to_pickle("build.pkl")
    train_weat_df.to_pickle("weather_train.pkl")
    infer_weat_df.to_pickle("weather_infer.pkl")
