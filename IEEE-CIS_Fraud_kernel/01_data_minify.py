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


# reduce memory for dataframe
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            # int8:-128~127; int16:-32768
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
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def minify_identity_df(df):
    df['id_12'] = df['id_12'].map({'Found': 1, 'NotFound': 0})
    df['id_15'] = df['id_15'].map({'New': 2, 'Found': 1, 'Unknown': 0})
    df['id_16'] = df['id_16'].map({'Found': 1, 'NotFound': 0})

    df['id_23'] = df['id_23'].map({'TRANSPARENT': 4, 'IP_PROXY': 3, 'IP_PROXY:ANONYMOUS': 2, 'IP_PROXY:HIDDEN': 1})

    df['id_27'] = df['id_27'].map({'Found': 1, 'NotFound': 0})
    df['id_28'] = df['id_28'].map({'New': 2, 'Found': 1})

    df['id_29'] = df['id_29'].map({'Found': 1, 'NotFound': 0})

    df['id_35'] = df['id_35'].map({'T': 1, 'F': 0})
    df['id_36'] = df['id_36'].map({'T': 1, 'F': 0})
    df['id_37'] = df['id_37'].map({'T': 1, 'F': 0})
    df['id_38'] = df['id_38'].map({'T': 1, 'F': 0})

    df['id_34'] = df['id_34'].fillna(':0')
    df['id_34'] = df['id_34'].apply(lambda x: x.split(':')[1]).astype(np.int8)
    df['id_34'] = np.where(df['id_34'] == 0, np.nan, df['id_34'])

    df['id_33'] = df['id_33'].fillna('0x0')
    df['id_33_0'] = df['id_33'].apply(lambda x: x.split('x')[0]).astype(int)
    df['id_33_1'] = df['id_33'].apply(lambda x: x.split('x')[1]).astype(int)
    df['id_33'] = np.where(df['id_33'] == '0x0', np.nan, df['id_33'])

    df['DeviceType'].map({'desktop': 1, 'mobile': 0})
    return df


if __name__ == "__main__":
    print("========== 1.Set random seed ...")
    SEED = 42
    set_seed(SEED)

    print("========== 2.Load csv data ...")
    LOCAL_TEST = False
    dir_data_csv = os.getcwd() + "\\ieee-fraud-detection\\"
    train_tran = pd.read_csv(dir_data_csv + "\\train_transaction.csv")
    train_iden = pd.read_csv(dir_data_csv + "\\train_identity.csv")
    infer_tran = pd.read_csv(dir_data_csv + "\\test_transaction.csv")
    infer_iden = pd.read_csv(dir_data_csv + "\\test_identity.csv")
    infer_tran["isFraud"] = 0

    if LOCAL_TEST:
        for _df in [train_tran, infer_tran, train_iden, infer_iden]:
            df1 = reduce_mem_usage(_df)
            for col1 in list(df1):
                if not df1[col1].equals(_df[col1]):
                    print("Bad transformation!!!", col1)

    train_df = reduce_mem_usage(train_tran)
    test_df = reduce_mem_usage(infer_tran)
    train_identity = reduce_mem_usage(train_iden)
    test_identity = reduce_mem_usage(infer_iden)
    for col in ['card4', 'card6', 'ProductCD']:
        print('Encoding', col)
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        col_encoded = temp_df[col].value_counts().to_dict()
        train_df[col] = train_df[col].map(col_encoded)
        test_df[col] = test_df[col].map(col_encoded)
        print(col_encoded)

    for col in ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']:
        train_df[col] = train_df[col].map({'T': 1, 'F': 0})
        test_df[col] = test_df[col].map({'T': 1, 'F': 0})

    for col in ['M4']:
        print('Encoding', col)
        temp_df = pd.concat([train_df[[col]], test_df[[col]]])
        col_encoded = temp_df[col].value_counts().to_dict()
        train_df[col] = train_df[col].map(col_encoded)
        test_df[col] = test_df[col].map(col_encoded)
        print(col_encoded)

    train_identity = minify_identity_df(train_identity)
    test_identity = minify_identity_df(test_identity)

    for col in ['id_33']:
        train_identity[col] = train_identity[col].fillna('unseen_before_label')
        test_identity[col] = test_identity[col].fillna('unseen_before_label')

        le = LabelEncoder()
        le.fit(list(train_identity[col]) + list(test_identity[col]))
        train_identity[col] = le.transform(train_identity[col])
        test_identity[col] = le.transform(test_identity[col])

    train_df = reduce_mem_usage(train_df)
    test_df = reduce_mem_usage(test_df)

    train_identity = reduce_mem_usage(train_identity)
    test_identity = reduce_mem_usage(test_identity)

    train_df.to_pickle('train_transaction.pkl')
    test_df.to_pickle('test_transaction.pkl')

    train_identity.to_pickle('train_identity.pkl')
    test_identity.to_pickle('test_identity.pkl')






