#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
############### TF Version: 1.13.1/Python Version: 3.7 ###############
"""

import os
import random
import datetime
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


if __name__ == "__main__":
    print("========== 1.Set random seed ... ==========")
    SEED = 42
    set_seed(SEED)
    START_DATE = datetime.datetime.strptime("2017-01-01", "%Y-%m-%d")

    print("========== 2.Load csv data ... ==========")
    dir_data_csv = os.getcwd() + "\\train\\"
    train_location = pd.read_csv(dir_data_csv + "\\站点信息.csv", encoding="gbk")
    train_loc = train_location.drop(labels=["UPDATE_TIME_", "SOURCE_"], axis=1)
    print(train_loc.shape)
    print(train_loc.columns.tolist())

    train_loc0 = pd.read_csv(dir_data_csv + "\\train_大石西路.csv")
    train_loc1 = pd.read_csv(dir_data_csv + "\\train_金泉两河.csv")
    train_loc2 = pd.read_csv(dir_data_csv + "\\train_君平街.csv")
    train_loc3 = pd.read_csv(dir_data_csv + "\\train_灵岩寺.csv")
    train_loc4 = pd.read_csv(dir_data_csv + "\\train_龙泉驿区政府.csv")
    train_loc5 = pd.read_csv(dir_data_csv + "\\train_三瓦窑.csv")
    train_loc6 = pd.read_csv(dir_data_csv + "\\train_沙河铺.csv")
    train_loc7 = pd.read_csv(dir_data_csv + "\\train_十里店.csv")
    train_total = pd.concat([train_loc0, train_loc1, train_loc2, train_loc3,
                             train_loc4, train_loc5, train_loc6, train_loc7])
    train_tol = train_total.drop(labels=["UPDATE_TIME_"], axis=1)
    print(train_tol.shape)
    print(train_tol.columns.tolist())

    train_df = train_tol.merge(train_loc, on=["MN_"], how="left")
    print(train_df.shape)
    print(train_df.columns.tolist())

    # 时间格式变换 DATA_TIME_
    train_df["DT"] = train_df["DATA_TIME_"].apply(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S"))

    # DATA_TIME_ 监测时间
    for df in [train_df]:
        df["DT_M"] = (df["DT"].dt.year - 2017) * 12 + df["DT"].dt.month
        df["DT_W"] = (df["DT"].dt.year - 2017) * 52 + df["DT"].dt.weekofyear
        df["DT_D"] = (df["DT"].dt.year - 2017) * 365 + df["DT"].dt.dayofyear

        df["DT_hour"] = df["DT"].dt.hour
        df["DT_day_week"] = df["DT"].dt.dayofweek
        df["DT_day"] = df["DT"].dt.day

    print(train_df.shape)
    print(train_df.columns.tolist())
    train_df.to_csv("ori_data.csv", sep=",", header=True, index=False)

