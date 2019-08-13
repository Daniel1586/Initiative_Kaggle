#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Data Analysis IEEE-CIS Fraud Detection dataset.
(https://www.kaggle.com/c/ieee-fraud-detection).
----数据解压: train.csv=45840617条样本[has label], test.csv=6042135条样本[no label]
----从train.txt取最后330000条数据,最后30000条数据为测试集;
----前面300000条数据按9:1比例随机选取为训练集/验证集;
----从test.txt取开始30000条数据为infer样本集;
This code is referenced from PaddlePaddle models.
(https://github.com/PaddlePaddle/models/blob/develop/legacy/deep_fm/preprocess.py)
--For numeric features, clipped and normalized.
--For categorical features, removed long-tailed data appearing less than 200 times.
############### TF Version: 1.13.1/Python Version: 3.7 ###############
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

input_dir = os.getcwd() + "\\ieee-fraud-detection"
print(os.listdir(input_dir))

# import data [index_col指定哪一列数据作为行索引,返回DataFrame]
train_tran = pd.read_csv(input_dir + "\\train_transaction.csv", index_col="TransactionID")
train_iden = pd.read_csv(input_dir + "\\train_identity.csv", index_col="TransactionID")
tests_tran = pd.read_csv(input_dir + "\\test_transaction.csv", index_col="TransactionID")
tests_iden = pd.read_csv(input_dir + "\\test_identity.csv", index_col="TransactionID")

train = train_tran.merge(train_iden, how="left", left_index=True, right_index=True)
print(train.shape)      # (590540, 433)
print(train.head(5))

tests = tests_tran.merge(tests_iden, how="left", left_index=True, right_index=True)
print(tests.shape)      # (506691, 432)
print(tests.head(5))

y_train = train["isFraud"].copy()
print(y_train.shape)    # (590540,)
print(y_train.head(5))
x_train = train.drop("isFraud", axis=1)
print(x_train.shape)    # (590540, 432)
print(x_train.head(5))
x_tests = tests.copy()
print(x_tests.shape)    # (506691, 432)

# explore data [describe single variables]
# Categorical => isFraud/ProductCD/DeviceType——Figure_1.png
# isFraud极不平衡[0/1],ProductCD不平衡[W/H/C/S/R],DeviceType=desktop:mobile=86:56
plt_show = 0
if plt_show:
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    isFraud = sns.countplot(x="isFraud", data=train, ax=axes[0])
    ProductCD = sns.countplot(x="ProductCD", data=train, ax=axes[1])
    DeviceType = sns.countplot(x="DeviceType", data=train, ax=axes[2])
    plt.tight_layout()
    plt.show()

# Categorical => DeviceInfo——Figure_2.png
# the top devices are: Windows|IOS Device|MacOS|Trident/7.0|...
plt_show = 0
if plt_show:
    group = pd.DataFrame()
    group["DeviceCount"] = train.groupby(["DeviceInfo"])["DeviceInfo"].count()
    group["DeviceInfo"] = group.index
    group_top = group.sort_values(by="DeviceCount", ascending=False).head(10)
    plt.figure(figsize=(15, 6))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    ax = sns.barplot(x="DeviceInfo", y="DeviceCount", data=group_top)
    xt = plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# Categorical => card1/card2/card3/card5——Figure_3.png
# card1/card2取值较多且较均匀,card3/card5取值较少且分布不平衡
plt_show = 0
if plt_show:
    _, axes = plt.subplots(4, 1, figsize=(20, 8))
    c1 = sns.distplot(train["card1"].dropna(), kde=False, ax=axes[0])
    c2 = sns.distplot(train["card2"].dropna(), kde=False, ax=axes[1])
    c3 = sns.distplot(train["card3"].dropna(), kde=False, ax=axes[2])
    c5 = sns.distplot(train["card5"].dropna(), kde=False, ax=axes[3])
    plt.tight_layout()
    # plt.show()

# Categorical => card4/card6——Figure_4.png
# card4绝大部分是visa/mastercard,card5绝大部分是debit/credit
plt_show = 0
if plt_show:
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    c4 = sns.countplot(x="card4", data=train, ax=axes[0])  # 样本极不平衡[0/1]
    c6 = sns.countplot(x="card6", data=train, ax=axes[1])  # 样本极不平衡[W/H/C/S/R]
    plt.tight_layout()
    plt.show()
