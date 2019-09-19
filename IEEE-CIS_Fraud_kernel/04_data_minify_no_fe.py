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
import random
import operator
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
    dir_data_csv = os.getcwd() + "\\ieee-fraud-detection\\"
    train_tran = pd.read_csv(dir_data_csv + "\\train_transaction.csv")
    train_iden = pd.read_csv(dir_data_csv + "\\train_identity.csv")
    infer_tran = pd.read_csv(dir_data_csv + "\\test_transaction.csv")
    infer_iden = pd.read_csv(dir_data_csv + "\\test_identity.csv")
    infer_tran["isFraud"] = 0
    if LOCAL_TEST:
        for df1 in [train_tran, infer_tran, train_iden, infer_iden]:
            df2 = reduce_mem_usage(df1)
            print("-----col num: ", len(list(df2)))
            for col1 in list(df2):
                if not df2[col1].equals(df1[col1]):
                    print("-----Bad transformation!!!", col1)

    print("========== 3.Optimize dataframe memory ...")
    train_df = reduce_mem_usage(train_tran)
    infer_df = reduce_mem_usage(infer_tran)
    train_id_df = reduce_mem_usage(train_iden)
    infer_id_df = reduce_mem_usage(infer_iden)

    print("========== 5.Save pkl ...")
    dir_data_pkl = os.getcwd() + "\\ieee-fraud-pkl-no-fe\\"
    train_df.to_pickle(dir_data_pkl + "\\train_tran_no_fe.pkl")
    infer_df.to_pickle(dir_data_pkl + "\\infer_tran_no_fe.pkl")
    train_id_df.to_pickle(dir_data_pkl + "\\train_iden_no_fe.pkl")
    infer_id_df.to_pickle(dir_data_pkl + "\\infer_iden_no_fe.pkl")
