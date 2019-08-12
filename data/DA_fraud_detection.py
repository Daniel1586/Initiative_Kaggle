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

# explore data

