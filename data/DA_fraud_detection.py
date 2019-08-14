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

# =============================================================================
# =============================================================================
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
# DeviceInfo取值数量1786
plt_show = 0
print(train["DeviceInfo"].nunique())
if plt_show:
    group = pd.DataFrame()
    group["DeviceCount"] = train.groupby(["DeviceInfo"])["DeviceInfo"].count()
    group["DeviceInfo"] = group.index
    group_top = group.sort_values(by="DeviceCount", ascending=False).head(10)
    plt.figure(figsize=(15, 6))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="DeviceInfo", y="DeviceCount", data=group_top)
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# Categorical => card1/card2/card3/card5——Figure_3.png
# card1/card2取值较多且较均匀,card3/card5取值较少且分布不平衡
# card1取值数量13553,card2取值数量500,card3取值数量114,card5取值数量119
plt_show = 0
print(train["card1"].nunique())
print(train["card2"].nunique())
print(train["card3"].nunique())
print(train["card5"].nunique())
if plt_show:
    _, axes = plt.subplots(4, 1, figsize=(20, 8))
    c1 = sns.distplot(train["card1"].dropna(), kde=False, ax=axes[0])
    c2 = sns.distplot(train["card2"].dropna(), kde=False, ax=axes[1])
    c3 = sns.distplot(train["card3"].dropna(), kde=False, ax=axes[2])
    c5 = sns.distplot(train["card5"].dropna(), kde=False, ax=axes[3])
    plt.tight_layout()
    plt.show()

# Categorical => card4/card6——Figure_4.png
# card4绝大部分是visa/mastercard,card5绝大部分是debit/credit
plt_show = 0
if plt_show:
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    c4 = sns.countplot(x="card4", data=train, ax=axes[0])
    c6 = sns.countplot(x="card6", data=train, ax=axes[1])
    plt.tight_layout()
    plt.show()

# Categorical => addr1——Figure_5.png
# addr1数据前20呈现近似对偶性,取值数量332
plt_show = 0
print(train["addr1"].nunique())
if plt_show:
    group = pd.DataFrame()
    group["addr1Count"] = train.groupby(["addr1"])["addr1"].count()
    group["addr1"] = group.index
    group_top = group.sort_values(by="addr1Count", ascending=False).head(20)
    plt.figure(figsize=(15, 6))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="addr1", y="addr1Count", data=group_top)
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# Categorical => addr2——Figure_6.png
# addr2存在1个点占近88%比例,取值数量74
plt_show = 0
print(train["addr2"].nunique())
print(train["addr2"].value_counts().head(10))
if plt_show:
    group = pd.DataFrame()
    group["addr2Count"] = train.groupby(["addr2"])["addr2"].count()
    group["addr2"] = group.index
    group_top = group.sort_values(by="addr2Count", ascending=False).head(20)
    plt.figure(figsize=(15, 6))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="addr2", y="addr2Count", data=group_top)
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# Categorical => P_emaildomain/R_emaildomain——Figure_7.png
# gmail占比最高,存在anonymous.com
# P_emaildomain取值数量59,R_emaildomain取值数量60
plt_show = 0
print(train["P_emaildomain"].nunique())
print(train["R_emaildomain"].nunique())
if plt_show:
    _, axes = plt.subplots(1, 2, figsize=(18, 9))
    sns.set(color_codes=True)
    p_email = sns.countplot(y="P_emaildomain", data=train, ax=axes[0])
    r_email = sns.countplot(y="R_emaildomain", data=train, ax=axes[1])
    plt.tight_layout()
    plt.show()

# Categorical => M1-M9——Figure_8.png
# M4取值数量为3[M0/M1/M2],其他取值数量为2[T/F]
plt_show = 0
m1_loc = train.columns.get_loc("M1")
m9_loc = train.columns.get_loc("M9")
df_m = train.iloc[:, m1_loc:m9_loc+1]
cols = df_m.columns
if plt_show:
    _, axes = plt.subplots(3, 3, figsize=(16, 12))
    count = 0
    for i in range(3):
        for j in range(3):
            m_plt = sns.countplot(x=cols[count], data=df_m, ax=axes[i, j])
            count += 1
    plt.tight_layout()
    plt.show()

# Categorical => id_12-id_38——Figure_9.png/Figure_10.png
# 存在混合数据类型,NaN比例较大,id30为OS,id31为浏览器
id12_loc = train.columns.get_loc("id_12")
id38_loc = train.columns.get_loc("id_38")
df_id = train.iloc[:, id12_loc:id38_loc+1]
print(df_id.dtypes)
print(df_id.head(15))

# id30取值数量75,id30为OS
plt_show = 0
print(train["id_30"].nunique())
if plt_show:
    group = pd.DataFrame()
    group["id_30Count"] = train.groupby(["id_30"])["id_30"].count()
    group["id_30"] = group.index
    group_top = group.sort_values(by="id_30Count", ascending=False).head(10)
    plt.figure(figsize=(15, 6))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="id_30", y="id_30Count", data=group_top)
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# id31取值数量130,id31为浏览器
plt_show = 0
print(train["id_31"].nunique())
if plt_show:
    group = pd.DataFrame()
    group["id_31Count"] = train.groupby(["id_31"])["id_31"].count()
    group["id_31"] = group.index
    group_top = group.sort_values(by="id_31Count", ascending=False).head(10)
    plt.figure(figsize=(15, 6))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="id_31", y="id_31Count", data=group_top)
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# =============================================================================
# =============================================================================
# explore data [describe continuous variables]
# Numeric => TransactionDT/TransactionAmt——Figure_11.png
# TransactionDT均匀分布,TransactionAmt指数分布[存在离群点]
plt_show = 0
if plt_show:
    _, axes = plt.subplots(2, 1, figsize=(15, 10))
    trans_dt = sns.distplot(train["TransactionDT"], kde=False, ax=axes[0])
    trans_amt = sns.distplot(train["TransactionAmt"], kde=False, ax=axes[1], hist_kws={"log": True})
    plt.tight_layout()
    plt.show()

# Numeric => C7-C14——Figure_12.png
# C7-C14近似指数分布
plt_show = 0
c7_loc = train.columns.get_loc("C7")
c14_loc = train.columns.get_loc("C14")
df_c = train.iloc[:, c7_loc:c14_loc+1]
cols = df_c.columns
if plt_show:
    _, axes = plt.subplots(4, 2, figsize=(15, 10))
    count = 0
    for i in range(4):
        for j in range(2):
            c_plt = sns.distplot(df_c[cols[count]], kde=False, hist_kws={"log": True}, ax=axes[i, j])
            count += 1
    plt.tight_layout()
    plt.show()

# Numeric => D1-D15——Figure_13.png
# D11-D15存在负值,D9特殊分布,其他近似指数分布
plt_show = 0
d1_loc = train.columns.get_loc("D1")
d15_loc = train.columns.get_loc("D15")
df_d = train.iloc[:, d1_loc:d15_loc+1]
cols = df_d.columns
if plt_show:
    _, axes = plt.subplots(5, 3, figsize=(15, 10))
    count = 0
    for i in range(5):
        for j in range(3):
            d_plt = sns.distplot(df_d[cols[count]].dropna(), ax=axes[i, j])
            count += 1
    plt.tight_layout()
    plt.show()

# Numeric => V1-V339
# V1-V305,V322-V339大部分取值为0/1,可能是categorical特征
# V306-V321 seems to be true continuous variables
v1_loc = train.columns.get_loc("V1")
v339_loc = train.columns.get_loc("V339")
df_v = train.iloc[:, v1_loc:v339_loc+1]
print(df_v.head(20))

# Numeric => id_01-id_06——Figure_14.png
# id02 may be dollar amounts, with log distribution
plt_show = 0
id01_loc = train.columns.get_loc("id_01")
id06_loc = train.columns.get_loc("id_06")
df1 = train.iloc[:, id01_loc:id06_loc+1]
cols = df1.columns
if plt_show:
    _, axes = plt.subplots(6, 2, figsize=(15, 10))
    for i in range(6):
        id1_plt = sns.distplot(df1[cols[i]].dropna(), ax=axes[i, 0])
        id2_plt = sns.distplot(df1[cols[i]].dropna(), kde=False, hist_kws={"log": True}, ax=axes[i, 1])
    plt.tight_layout()
    # plt.show()

# Numeric => id_07-id_11——Figure_15.png
# id07/id08近似正态分布
plt_show = 0
id07_loc = train.columns.get_loc("id_07")
id11_loc = train.columns.get_loc("id_11")
df2 = train.iloc[:, id07_loc:id11_loc+1]
cols = df2.columns
if plt_show:
    _, axes = plt.subplots(5, 2, figsize=(15, 10))
    for i in range(5):
        id1_plt = sns.distplot(df2[cols[i]].dropna(), ax=axes[i, 0])
        id2_plt = sns.distplot(df2[cols[i]].dropna(), kde=False, hist_kws={"log": True}, ax=axes[i, 1])
    plt.tight_layout()
    plt.show()
