#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Data Analysis IEEE-CIS Fraud Detection dataset.
(https://www.kaggle.com/c/ieee-fraud-detection).
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
# ProductCD: W/C类别欺诈样本数量最多, C/S类别欺诈比例最高
# DeviceType: mobile/desktop欺诈样本数量接近,但mobile类别欺诈比例较高
plt_show = 0
if plt_show:
    _, axes = plt.subplots(3, 3, figsize=(16, 9))
    # 原始分布
    isFraud = sns.countplot(x="isFraud", data=train, ax=axes[0, 0])
    ProductCD = sns.countplot(x="ProductCD", data=train, ax=axes[0, 1])
    DeviceType = sns.countplot(x="DeviceType", data=train, ax=axes[0, 2])
    # 带target的分布
    isFraud1 = sns.countplot(x="isFraud", data=train, ax=axes[1, 0])
    ProductCD1 = sns.countplot(x="ProductCD", hue="isFraud", data=train, ax=axes[1, 1])
    DeviceType1 = sns.countplot(x="DeviceType", hue="isFraud", data=train, ax=axes[1, 2])
    # 带target的百分比分布
    isFraud2 = sns.countplot(x="isFraud", data=train, ax=axes[2, 0])
    props1 = train.groupby("ProductCD")["isFraud"].value_counts(normalize=True).unstack()
    props1.plot(kind="bar", stacked="True", ax=axes[2, 1])
    props2 = train.groupby("DeviceType")["isFraud"].value_counts(normalize=True).unstack()
    props2.plot(kind="bar", stacked="True", ax=axes[2, 2])
    plt.tight_layout()
    plt.show()

# Categorical => DeviceInfo——Figure_2.png
# the top devices are: Windows|IOS Device|MacOS|Trident/7.0|占据90%以上
# DeviceInfo取值数量1786
# 欺诈样本中DeviceInfo为Windows/IOS Device占据75%以上
plt_show = 0
if plt_show:
    print(train["DeviceInfo"].nunique())
    group = pd.DataFrame()
    group["DeviceCount"] = train.groupby(["DeviceInfo"])["DeviceInfo"].count()
    group["DeviceInfo"] = group.index
    group_top = group.sort_values(by="DeviceCount", ascending=False).head(10)

    fraud = pd.DataFrame()
    is_fraud = train[train["isFraud"] == 1]
    fraud["DeviceCount"] = is_fraud.groupby(["DeviceInfo"])["DeviceInfo"].count()
    fraud["DeviceInfo"] = fraud.index
    fraud_top = fraud.sort_values(by="DeviceCount", ascending=False).head(10)

    _, axes = plt.subplots(2, 1, figsize=(16, 9))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="DeviceInfo", y="DeviceCount", data=group_top, ax=axes[0])
    ax = sns.barplot(x="DeviceInfo", y="DeviceCount", data=fraud_top, ax=axes[1])
    ax.set_title("Fraud transactions by DeviceInfo")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# Categorical => card1/card2/card3/card5——Figure_3_1.png/Figure_3_2.png
# card1取值数量13553,card2取值数量500,card3取值数量114,card5取值数量119
# card1/card2取值较多且较均匀,card3/card5取值较少且分布不平衡
# card1/card2/card3在欺诈/非欺诈样本中分布类似,card5欺诈/非欺诈样本在值225左右存在较大差异
plt_show = 0
if plt_show:
    print(train["card1"].nunique())
    print(train["card2"].nunique())
    print(train["card3"].nunique())
    print(train["card5"].nunique())
    _, axes1 = plt.subplots(4, 1, figsize=(16, 9))
    c1 = sns.distplot(train["card1"].dropna(), kde=False, ax=axes1[0])
    c2 = sns.distplot(train["card2"].dropna(), kde=False, ax=axes1[1])
    c3 = sns.distplot(train["card3"].dropna(), kde=False, ax=axes1[2])
    c5 = sns.distplot(train["card5"].dropna(), kde=False, ax=axes1[3])
    plt.tight_layout()

    is_fraud = train[train["isFraud"] == 1]
    no_fraud = train[train["isFraud"] == 0]
    _, axes2 = plt.subplots(4, 1, figsize=(16, 9))
    no_1 = sns.distplot(no_fraud["card1"].dropna(), color="fuchsia", label="No fraud", ax=axes2[0])
    l11 = no_1.legend()
    is_1 = sns.distplot(is_fraud["card1"].dropna(), color="black", label="Is Fraud", ax=axes2[0])
    l12 = is_1.legend()

    no_2 = sns.distplot(no_fraud["card2"].dropna(), color="fuchsia", label="No fraud", ax=axes2[1])
    l21 = no_2.legend()
    is_2 = sns.distplot(is_fraud["card2"].dropna(), color="black", label="Is Fraud", ax=axes2[1])
    l22 = is_2.legend()

    no_3 = sns.distplot(no_fraud["card3"].dropna(), color="fuchsia", label="No fraud", ax=axes2[2])
    l31 = no_3.legend()
    is_3 = sns.distplot(is_fraud["card3"].dropna(), color="black", label="Is Fraud", ax=axes2[2])
    l32 = is_3.legend()

    no_5 = sns.distplot(no_fraud["card5"].dropna(), color="fuchsia", label="No fraud", ax=axes2[3])
    l51 = no_5.legend()
    is_5 = sns.distplot(is_fraud["card5"].dropna(), color="black", label="Is Fraud", ax=axes2[3])
    l52 = is_5.legend()
    plt.tight_layout()
    plt.show()

# Categorical => card4/card6——Figure_4.png
# card4绝大部分是visa/mastercard,card5绝大部分是debit/credit
# card4样本取值visa欺诈数量最多,但比例较小;取值discover欺诈数量最少,但欺诈比例最高
# card5样本取值debit/credit占欺诈数量绝大多数,但credit欺诈比例最高
plt_show = 0
if plt_show:
    _, axes = plt.subplots(3, 2, figsize=(16, 9))
    # 原始分布
    c4_1 = sns.countplot(x="card4", data=train, ax=axes[0, 0])
    c6_1 = sns.countplot(x="card6", data=train, ax=axes[0, 1])
    # 带target的分布
    c4_2 = sns.countplot(x="card4", hue="isFraud", data=train, ax=axes[1, 0])
    c6_2 = sns.countplot(x="card6", hue="isFraud", data=train, ax=axes[1, 1])
    # 带target的百分比分布
    props1 = train.groupby("card4")["isFraud"].value_counts(normalize=True).unstack()
    props1.plot(kind="bar", stacked="True", ax=axes[2, 0])
    props2 = train.groupby("card6")["isFraud"].value_counts(normalize=True).unstack()
    props2.plot(kind="bar", stacked="True", ax=axes[2, 1])
    plt.tight_layout()
    plt.show()

# Categorical => addr1——Figure_5_1.png/Figure_5_2.png
# addr1取值数量332,数据前20呈现近似对偶性
# 欺诈样本最多的addr1取值为204/325/299,非欺诈样本最多的addr1取值为299/325/204
# 欺诈样本和非欺诈样本前5取值一样,只是数量顺序有差异
plt_show = 0
if plt_show:
    print(train["addr1"].nunique())
    group = pd.DataFrame()
    group["addr1Count"] = train.groupby(["addr1"])["addr1"].count()
    group["addr1"] = group.index
    group_top = group.sort_values(by="addr1Count", ascending=False).head(20)
    plt.figure(figsize=(16, 9))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="addr1", y="addr1Count", data=group_top)
    plt.xticks(rotation=60)
    plt.tight_layout()

    addr1_is = pd.DataFrame()
    is_fraud = train[train["isFraud"] == 1]
    addr1_is["addr1Count"] = is_fraud.groupby(["addr1"])["addr1"].count()
    addr1_is["addr1"] = addr1_is.index

    addr1_no = pd.DataFrame()
    no_fraud = train[train["isFraud"] == 0]
    addr1_no["addr1Count"] = no_fraud.groupby(["addr1"])["addr1"].count()
    addr1_no["addr1"] = addr1_no.index

    group_top_f = addr1_is.sort_values(by="addr1Count", ascending=False).head(20)
    order_f = group_top_f.sort_values(by="addr1Count", ascending=False)["addr1"]
    group_top_l = addr1_no.sort_values(by="addr1Count", ascending=False).head(20)
    order_l = group_top_l.sort_values(by="addr1Count", ascending=False)["addr1"]

    _, axes = plt.subplots(4, 1, figsize=(16, 9))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    ax = sns.barplot(x="addr1", y="addr1Count", data=group_top_f, order=order_f, ax=axes[0])
    bx = sns.barplot(x="addr1", y="addr1Count", data=group_top_l, order=order_l, ax=axes[1])
    az = sns.barplot(x="addr1", y="addr1Count", data=group_top_f, ax=axes[2])
    bz = sns.barplot(x="addr1", y="addr1Count", data=group_top_l, ax=axes[3])
    ax.set_title("Fraud transactions by addr1 (ranked)")
    bx.set_title("Legit transactions by addr1 (ranked)")
    az.set_title("Fraud transactions by addr1")
    bz.set_title("Legit transactions by addr1")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# Categorical => addr2——Figure_6.png
# addr2存在1个点占近88%比例[取值极不平衡],取值数量74
plt_show = 0
if plt_show:
    print(train["addr2"].nunique())
    print(train["addr2"].value_counts().head(10))
    group = pd.DataFrame()
    group["addr2Count"] = train.groupby(["addr2"])["addr2"].count()
    group["addr2"] = group.index
    group_top = group.sort_values(by="addr2Count", ascending=False).head(20)
    plt.figure(figsize=(16, 9))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    sns.barplot(x="addr2", y="addr2Count", data=group_top)
    plt.xticks(rotation=60)
    plt.tight_layout()

    addr2_is = pd.DataFrame()
    is_fraud = train[train["isFraud"] == 1]
    addr2_is["addr2Count"] = is_fraud.groupby(["addr2"])["addr2"].count()
    addr2_is["addr2"] = addr2_is.index

    addr2_no = pd.DataFrame()
    no_fraud = train[train["isFraud"] == 0]
    addr2_no["addr2Count"] = no_fraud.groupby(["addr2"])["addr2"].count()
    addr2_no["addr2"] = addr2_no.index

    group_top_f = addr2_is.sort_values(by="addr2Count", ascending=False).head(20)
    order_f = group_top_f.sort_values(by="addr2Count", ascending=False)["addr2"]
    group_top_l = addr2_no.sort_values(by="addr2Count", ascending=False).head(20)
    order_l = group_top_l.sort_values(by="addr2Count", ascending=False)["addr2"]

    _, axes = plt.subplots(4, 1, figsize=(16, 9))
    sns.set(color_codes=True)
    sns.set(font_scale=1.3)
    ax = sns.barplot(x="addr2", y="addr2Count", data=group_top_f, order=order_f, ax=axes[0])
    bx = sns.barplot(x="addr2", y="addr2Count", data=group_top_l, order=order_l, ax=axes[1])
    az = sns.barplot(x="addr2", y="addr2Count", data=group_top_f, ax=axes[2])
    bz = sns.barplot(x="addr2", y="addr2Count", data=group_top_l, ax=axes[3])
    ax.set_title("Fraud transactions by addr2 (ranked)")
    bx.set_title("Legit transactions by addr2 (ranked)")
    az.set_title("Fraud transactions by addr2")
    bz.set_title("Legit transactions by addr2")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

# Categorical => P_emaildomain/R_emaildomain——Figure_7.png
# gmail占比最高,存在anonymous.com
# P_emaildomain取值数量59,R_emaildomain取值数量60
plt_show = 1
if plt_show:
    print(train["P_emaildomain"].nunique())
    print(train["R_emaildomain"].nunique())
    _, axes1 = plt.subplots(1, 2, figsize=(16, 9))
    sns.set(color_codes=True)
    p_email = sns.countplot(y="P_emaildomain", data=train, ax=axes1[0])
    r_email = sns.countplot(y="R_emaildomain", data=train, ax=axes1[1])
    plt.tight_layout()

    order_p = train["P_emaildomain"].value_counts().iloc[:10].index
    order_r = train["R_emaildomain"].value_counts().iloc[:10].index
    _, axes2 = plt.subplots(1, 2, figsize=(16, 9))
    sns.set(color_codes=True)
    sns.countplot(y="P_emaildomain", hue="isFraud", data=train, order=order_p, ax=axes2[0])
    sns.countplot(y="R_emaildomain", hue="isFraud", data=train, order=order_r, ax=axes2[1])
    plt.tight_layout()
    plt.show()

# Categorical => M1-M9——Figure_8.png
# M4取值数量为3[M0/M1/M2],其他取值数量为2[T/F]
plt_show = 0
if plt_show:
    m1_loc = train.columns.get_loc("M1")
    m9_loc = train.columns.get_loc("M9")
    df_m = train.iloc[:, m1_loc:m9_loc + 1]
    cols = df_m.columns
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
plt_show = 0
if plt_show:
    id12_loc = train.columns.get_loc("id_12")
    id38_loc = train.columns.get_loc("id_38")
    df_id = train.iloc[:, id12_loc:id38_loc + 1]
    print(df_id.dtypes)
    print(df_id.head(15))

# id30取值数量75,id30为OS
plt_show = 0
if plt_show:
    print(train["id_30"].nunique())
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
if plt_show:
    print(train["id_31"].nunique())
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
if plt_show:
    c7_loc = train.columns.get_loc("C7")
    c14_loc = train.columns.get_loc("C14")
    df_c = train.iloc[:, c7_loc:c14_loc + 1]
    cols = df_c.columns
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
if plt_show:
    d1_loc = train.columns.get_loc("D1")
    d15_loc = train.columns.get_loc("D15")
    df_d = train.iloc[:, d1_loc:d15_loc + 1]
    cols = df_d.columns
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
plt_show = 0
if plt_show:
    v1_loc = train.columns.get_loc("V1")
    v339_loc = train.columns.get_loc("V339")
    df_v = train.iloc[:, v1_loc:v339_loc + 1]
    print(df_v.head(20))

# Numeric => id_01-id_06——Figure_14.png
# id02 may be dollar amounts, with log distribution
plt_show = 0
if plt_show:
    id01_loc = train.columns.get_loc("id_01")
    id06_loc = train.columns.get_loc("id_06")
    df1 = train.iloc[:, id01_loc:id06_loc + 1]
    cols = df1.columns
    _, axes = plt.subplots(6, 2, figsize=(15, 10))
    for i in range(6):
        id1_plt = sns.distplot(df1[cols[i]].dropna(), ax=axes[i, 0])
        id2_plt = sns.distplot(df1[cols[i]].dropna(), kde=False, hist_kws={"log": True}, ax=axes[i, 1])
    plt.tight_layout()
    # plt.show()

# Numeric => id_07-id_11——Figure_15.png
# id07/id08近似正态分布
plt_show = 0
if plt_show:
    id07_loc = train.columns.get_loc("id_07")
    id11_loc = train.columns.get_loc("id_11")
    df2 = train.iloc[:, id07_loc:id11_loc + 1]
    cols = df2.columns
    _, axes = plt.subplots(5, 2, figsize=(15, 10))
    for i in range(5):
        id1_plt = sns.distplot(df2[cols[i]].dropna(), ax=axes[i, 0])
        id2_plt = sns.distplot(df2[cols[i]].dropna(), kde=False, hist_kws={"log": True}, ax=axes[i, 1])
    plt.tight_layout()
    plt.show()
