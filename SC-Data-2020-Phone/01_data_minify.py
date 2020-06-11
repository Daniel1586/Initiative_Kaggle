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

    print("========== 2.Load csv data ... ==========")
    dir_train = os.getcwd() + "\\train\\"
    dir_tests = os.getcwd() + "\\test\\"
    # 基础资料-----train_user/test_user
    train_user = pd.read_csv(dir_train + "\\train_user.csv")
    train_user__ = train_user.drop_duplicates(subset=["phone_no_m"], keep="first", inplace=False)
    print("----- train_user 大小:", train_user.shape)
    print("----- train_user 列名:", train_user.columns.tolist())
    print("----- train_user 字段 phone_no_m 无重复!", train_user__.shape)
    print("----- train_user 0/1标签数量\n", train_user.label.value_counts())
    tr_user = set(list(train_user__.phone_no_m))

    test_user = pd.read_csv(dir_tests + "\\test_user.csv")
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

    # labelEncoder标准化标签
    for col1 in ["city_name", "county_name"]:
        train_user[col1] = train_user[col1].fillna("UNK")
        test_user[col1] = test_user[col1].fillna("UNK")
        le = LabelEncoder()
        le.fit(list(train_user[col1]) + list(test_user[col1]))
        train_user[col1] = le.transform(train_user[col1])
        test_user[col1] = le.transform(test_user[col1])

    # 基础资料-----train_user/test_user
    # train_voc_ = pd.read_csv(dir_data_csv + "\\train_voc.csv")
    # print(train_voc_.shape)
    # train_sms_ = pd.read_csv(dir_data_csv + "\\train_sms.csv")
    # print(train_sms_.shape)
    # train_app_ = pd.read_csv(dir_data_csv + "\\train_app.csv")
    # print(train_app_.shape)

    infer_tran = pd.read_csv(dir_data_csv + "\\test_transaction.csv")
    infer_iden = pd.read_csv(dir_data_csv + "\\test_identity.csv")
    infer_tran["isFraud"] = 0

    print("========== 3.Optimize dataframe memory ...")
    train_df = reduce_mem_usage(train_tran)
    infer_df = reduce_mem_usage(infer_tran)
    # train_id_df = reduce_mem_usage(train_iden)
    # infer_id_df = reduce_mem_usage(infer_iden)

    print("========== 4.ETL tran categorical feature [String] ...")
    # [train+infer]离散特征(<=5): String->Int,字符串映射到数值[NaN不编码]-----num=12
    # ProductCD[W,C,R,H,S]//card4[visa,..]//card6[debit,..]//M1[T,F]//M2[T,F]
    # M3[T,F]//M4[M0,M1,M2]//M5[T,F]//M6[T,F]//M7[T,F]//M8[T,F]//M9[T,F]
    for col1 in ["ProductCD", "card4", "card6", "M1"]:
        total_df = pd.concat([train_df[[col1]], infer_df[[col1]]])
        tol_cols = total_df[col1].value_counts()
        print(tol_cols)

    for col1 in ["card4", "card6", "ProductCD", "M4"]:
        print("-----Encoding", col1)
        print(train_df[col1].value_counts())
        temp_df = pd.concat([train_df[[col1]], infer_df[[col1]]])
        col_encoded = temp_df[col1].value_counts().to_dict()
        train_df[col1] = train_df[col1].map(col_encoded)
        infer_df[col1] = infer_df[col1].map(col_encoded)
        print(train_df[col1].value_counts())
        print(col_encoded)

    # [train+infer]离散特征(=2):二值编码[NaN不编码]
    for col1 in ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]:
        train_df[col1] = train_df[col1].map({"T": 1, "F": 0})
        infer_df[col1] = infer_df[col1].map({"T": 1, "F": 0})

    # [train+infer]离散特征编码:[NaN不编码]
    train_id_df = minify_identity_df(train_id_df)
    infer_id_df = minify_identity_df(infer_id_df)

    # labelEncoder标准化标签
    for col1 in ['id_33']:
        train_id_df[col1] = train_id_df[col1].fillna('unseen_before_label')
        infer_id_df[col1] = infer_id_df[col1].fillna('unseen_before_label')

        le = LabelEncoder()
        le.fit(list(train_id_df[col1]) + list(infer_id_df[col1]))
        train_id_df[col1] = le.transform(train_id_df[col1])
        infer_id_df[col1] = le.transform(infer_id_df[col1])

    train_df = reduce_mem_usage(train_df)
    infer_df = reduce_mem_usage(infer_df)
    train_id_df = reduce_mem_usage(train_id_df)
    infer_id_df = reduce_mem_usage(infer_id_df)

    print("========== 5.Save pkl ...")
    train_df.to_pickle("train_transaction.pkl")
    infer_df.to_pickle("infer_transaction.pkl")
    train_id_df.to_pickle("train_identity.pkl")
    infer_id_df.to_pickle("infer_identity.pkl")
