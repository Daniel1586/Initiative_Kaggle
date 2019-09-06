#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Preprocess ieee-fraud-detection dataset.
(https://www.kaggle.com/c/ieee-fraud-detection).
--For numeric features, clipped and normalized.
--For categorical features, removed long-tailed data appearing less than 200 times.
############### TF Version: 1.13.1/Python Version: 3.7 ###############
"""

import os
import sys
import math
import random
import argparse
import collections
import pandas as pd

"""
Train shape:(590540,394),identity(144233,41)--isFraud 3.5%
Test  shape:(506691,393),identity(141907,41)
"""

numeric_features_etl = range(1, 33)
categorical_features = range(33, 82)
numeric_clip = [500, 900, 1000, 25, 25, 10, 10, 10, 20, 10, 10, 10, 10, 20,
                20, 100, 20, 500, 550, 150, 550, 250, 400, 300, 700, 1, 550,
                550, 400, 100, 400, 550]


def csv2txt_eda(datain_dir):
    # import data [index_col指定哪一列数据作为行索引,返回DataFrame]
    train_tran = pd.read_csv(datain_dir + "\\train_transaction.csv", index_col="TransactionID")
    train_iden = pd.read_csv(datain_dir + "\\train_identity.csv", index_col="TransactionID")
    train = train_tran.merge(train_iden, how="left", left_index=True, right_index=True)
    df_v = train["V38"]
    print(df_v.count())
    print(df_v.value_counts())
    print(df_v.value_counts()/df_v.count())
    print(df_v.min(), df_v.max())
    df_v1 = train[train["V38"] < 500]
    df_vv = df_v1["V38"]
    print(df_vv.count(), df_vv.count()/df_v.count())
    df_v2 = train[train["isFraud"] == 1]
    df_vvv = df_v2["V38"]
    print(df_vvv.value_counts())


def csv2txt(datain_dir, dataou_dir):
    # import data [index_col指定哪一列数据作为行索引,返回DataFrame]
    train_tran = pd.read_csv(datain_dir + "\\train_transaction.csv", index_col="TransactionID")
    train_iden = pd.read_csv(datain_dir + "\\train_identity.csv", index_col="TransactionID")
    tests_tran = pd.read_csv(datain_dir + "\\test_transaction.csv", index_col="TransactionID")
    tests_iden = pd.read_csv(datain_dir + "\\test_identity.csv", index_col="TransactionID")
    train = train_tran.merge(train_iden, how="left", left_index=True, right_index=True)
    tests = tests_tran.merge(tests_iden, how="left", left_index=True, right_index=True)

    # ==========num数值特征==========
    """
    TransactionAmt[0.251,31937.391]:valid=100%,unique=20902-->[0,500)=95.8%
    dist1------[0,10286]:valid=40.35%,unique=2652--->null=59.65%,[0,900)=38.45%-----[null用0.0替换??]
    dist2------[0,11623]:valid= 6.37%,unique=1752--->null=93.63%,[0,1000)=6.05%-----[null用0.0替换??]
    C1----------[0,4685]:valid=100.0%,unique=1657--->[0,25)=95.1%-------[corr>0.96: C11,C2,C6,C4,C8]
    C2----------[0,5691]:valid=100.0%,unique=1216--->[0,25)=94.9%-------[corr>0.97: C1,C11,C8,C6,C4]
    C3------------[0,26]:valid=100.0%,unique=27----->[0,10)=99.9%-------[corr>0.54: V28,V89,V68,V27]
    C4----------[0,2253]:valid=100.0%,unique=1260--->[0,10)=99.2%-------[corr>0.95: C11,C2,C1,C6,C8]
    C5-----------[0,349]:valid=100.0%,unique=349---->[0,10)=95.0%-------[corr>0.71: C9,C13]
    C6----------[0,2253]:valid=100.0%,unique=1328--->[0,20)=95.3%-------[corr>0.96: C11,C14,C1,C2,C4]
    C7----------[0,2255]:valid=100.0%,unique=1103--->[0,10)=99.4%-------[corr>0.92: C12,C10,C8,C2,C1]
    C8----------[0,3331]:valid=100.0%,unique=1253--->[0,10)=98.8%-------[corr>0.96: C10,C12,C7,C2,C1]
    C9-----------[0,210]:valid=100.0%,unique=205---->[0,10)=95.1%-------[corr>0.70: C5,C13]
    C10---------[0,3257]:valid=100.0%,unique=1231--->[0,10)=98.7%-------[corr>0.95: C8,C7,C12,C2,C1]
    C11---------[0,3188]:valid=100.0%,unique=1476--->[0,20)=95.3%-------[corr>0.96: C1,C2,C6,C4,C8]
    C12---------[0,3188]:valid=100.0%,unique=1199--->[0,20)=99.3%-------[corr>0.92: C7,C10,C8,C2,C1]
    C13---------[0,2918]:valid=100.0%,unique=1597--->[0,100)=94.9%------[corr>0.75: C14,C6,C11,C1,C2]
    C14---------[0,1429]:valid=100.0%,unique=1108--->[0,20)=95.4%-------[corr>0.90: C6,C11,C1,C2,C4]
    """
    # ==========cat离散特征==========
    """
    ProductCD[W,C,R,H,S]:valid=100.0%,unique=5----->W=74.45%,C=11.60%,R=6.38%,.
    card1---[1000,18396]:valid=100.0%,unique=13553->类均匀分布,7919=2.52%,9500=2.39%,.
    card2------[100,600]:valid=98.48%,unique=501--->类均匀分布,321=8.28%,111=7.65%,555=7.11%,.
    card3------[100,231]:valid=99.73%,unique=115--->150=88.27%,185=9.54%,.-----[corr>0.7: V15,V94,V16,V57,V79]
    card4-------[visa,.]:valid=99.73%,unique=5----->visa=65.15%,mastercard=32.04%,.
    card5------[100,237]:valid=99.27%,unique=120--->226=50.21%,224=13.80%,166=9.67%,.
    card6------[debit,.]:valid=99.73%,unique=5----->debit=74.49%,credit=25.22%,.
    addr1------[100,540]:valid=88.87%,unique=333--->类均匀分布,null=11.12%,299=7.84%,.
    addr2-------[10,102]:valid=88.87%,unique=75---->87=88.13%,null=11.12%,.
    P_emaildomain[.com,]:valid=84.00%,unique=60---->gmail.com=38.66%,yahoo.com=17.09%,null=15.99%,.
    R_emaildomain[.com,]:valid=23.25%,unique=61---->gmail.com=9.67%,hotmail.com=4.65%,anonymous.com=3.47%,.
    M1-------------[T,F]:valid=54.09%,unique=3----->T=54.08%,null=45.90%,.
    M2-------------[T,F]:valid=54.09%,unique=3----->T=48.34%,null=45.90%,.
    M3-------------[T,F]:valid=54.09%,unique=3----->null=45.90%,T=42.62%,.
    M4--------[M0,M1,M2]:valid=52.34%,unique=4----->null=47.65%,M0=33.25%,M2=10.13%,.
    M5-------------[T,F]:valid=40.65%,unique=3----->null=59.35%,F=22.43%,T=18.22%
    M6-------------[T,F]:valid=71.32%,unique=3----->F=38.58%,T=32.73,null=28.68%
    M7-------------[T,F]:valid=41.36%,unique=3----->null=58.63%,F=35.79%,T=5.57%
    M8-------------[T,F]:valid=41.36%,unique=3----->null=58.63%,F=26.29%,T=15.07%
    M9-------------[T,F]:valid=41.36%,unique=3----->null=58.63%,F=34.82%,T=6.54%
    """

    order = ["isFraud", "TransactionAmt", "dist1", "dist2", "C1", "C2", "C3", "C4", "C5",
             "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "D1", "D2", "D3",
             "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15",

             "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1",
             "addr2", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6",
             "M7", "M8", "M9", "id_12", "id_13", "id_14", "id_15", "id_16", "id_17", "id_18",
             "id_19", "id_20", "id_21", "id_22", "id_23", "id_24", "id_25", "id_26", "id_27",
             "id_28", "id_29", "id_30", "id_31", "id_32", "id_33", "id_34", "id_35", "id_36",
             "id_37", "id_38", "DeviceType", "DeviceInfo"]
    order_ = order[1:]
    train_txt = train[order]
    tests_txt = tests[order_]
    train_txt.to_csv(dataou_dir + "train.txt", sep='\t', index=False, header=0)
    tests_txt.to_csv(dataou_dir + "tests.txt", sep='\t', index=False, header=0)
    tests_idx = tests["ProductCD"]
    tests_idx.to_csv(dataou_dir + "index.csv", index=True, header=True)


def csv2csv(datain_dir, dataou_dir):
    idx = pd.read_csv(datain_dir + "\\index.csv")
    res = pd.read_csv(dataou_dir + "\\infer.csv", header=None, names=["isFraud"])
    tests = idx.merge(res, how="left", left_index=True, right_index=True)
    tests_txt = tests.drop("ProductCD", axis=1)
    tests_txt.to_csv(dataou_dir + "090101_DeepFM.csv", index=False, header=True)


class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorical_feature, cutoff=0):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                # 遍历离散特征,统计不同离散特征值出现次数
                for i in range(0, self.num_feature):
                    if features[categorical_feature[i]] != '':
                        self.dicts[i][features[categorical_feature[i]]] += 1

        for j in range(0, self.num_feature):
            # 剔除频次小于cutoff的离散特征,剩下特征按频次从大到小排序
            temp_list = filter(lambda x: x[1] >= cutoff, self.dicts[j].items())
            sort_list = sorted(temp_list, key=lambda x: (-x[1], x[0]))
            # 符合条件的离散特征, 编号1:len()-1, 不符合条件的特征编号为0
            vocabs, _ = list(zip(*sort_list))
            tran_dict = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[j] = tran_dict
            self.dicts[j]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return map(len, self.dicts)


class NumericFeatureGenerator:
    """
    Normalize the numeric features to [0, 1] by min-max normalization
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, numeric_feature):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    val = features[numeric_feature[i]]
                    if val != '':
                        val = float(val)
                        if val > numeric_clip[i]:
                            val = numeric_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        if self.min[idx] >= 0:
            _min = math.ceil(self.min[idx])
        else:
            _min = math.floor(self.min[idx])
        if self.max[idx] >= 0:
            _max = math.ceil(self.max[idx])
        else:
            _max = math.floor(self.max[idx])
        return (val - _min) / (_max - _min)


def preprocess(datain_dir, dataou_dir):
    print("========== 1.Preprocess categorical and numeric features...")
    c_feat = CategoryDictGenerator(len(categorical_features))
    c_feat.build(datain_dir + "train.txt", categorical_features, cutoff=FLAGS.cut_off)
    n_feat = NumericFeatureGenerator(len(numeric_features_etl))
    n_feat.build(datain_dir + "train.txt", numeric_features_etl)

    print("========== 2.Generate index of feature embedding ...")
    # 生成数值特征编号
    output = open(dataou_dir + "embed.set", 'w')
    for i in numeric_features_etl:
        output.write("{0} {1}\n".format('I'+str(i), i))

    dict_sizes = list(c_feat.dicts_sizes())
    c_feat_offset = [n_feat.num_feature]
    # 生成离散特征编号: C1|xx XX (不同离散特征第一个特征编号的特征统一为<unk>)
    for i in range(1, len(categorical_features)+1):
        offset = c_feat_offset[i - 1] + dict_sizes[i - 1]
        c_feat_offset.append(offset)
        for key, val in c_feat.dicts[i-1].items():
            output.write("{0} {1}\n".format('C'+str(i)+'|'+key, c_feat_offset[i - 1]+val+1))

    random.seed(0)
    # 90% data are used for training, and 10% data are used for validation
    print("========== 3.Generate train dataset ...")
    with open(dataou_dir + "train.set", 'w') as out_test:
        with open(datain_dir + "train.txt", 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_val = []
                for i in range(0, len(numeric_features_etl)):
                    val = n_feat.gen(i, features[numeric_features_etl[i]])
                    feat_val.append(str(numeric_features_etl[i]) + ':' + "{0:.6f}".format(val).
                                    rstrip('0').rstrip('.'))
                for i in range(0, len(categorical_features)):
                    val = c_feat.gen(i, features[categorical_features[i]]) + c_feat_offset[i] + 1
                    feat_val.append(str(val) + ':1')

                label = features[0]
                out_test.write("{0} {1}\n".format(label, ' '.join(feat_val)))

    print("========== 4.Generate infer dataset ...")
    with open(dataou_dir + "infer.set", 'w') as out_infer:
        with open(datain_dir + "tests.txt", 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_val = []
                for i in range(0, len(numeric_features_etl)):
                    val = n_feat.gen(i, features[numeric_features_etl[i] - 1])
                    feat_val.append(str(numeric_features_etl[i]) + ':' + "{0:.6f}".format(val).
                                    rstrip('0').rstrip('.'))
                for i in range(0, len(categorical_features)):
                    val = c_feat.gen(i, features[categorical_features[i] - 1]) + c_feat_offset[i] + 1
                    feat_val.append(str(val) + ':1')

                label = 0       # test fake label
                out_infer.write("{0} {1}\n".format(label, ' '.join(feat_val)))


if __name__ == "__main__":
    run_mode = 0        # 0: windows环境
    if run_mode == 0:
        dir_data_csv = os.getcwd() + "\\ieee-fraud-detection\\"
        dir_data_txt = os.getcwd() + "\\ieee-fraud-detection-txt\\"
        dir_data_set = os.getcwd() + "\\ieee-fraud-detection-set\\"
    else:
        dir_data_csv = ""
        dir_data_txt = ""
        dir_data_set = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4, help="threads num")
    parser.add_argument("--data_csv", type=str, default=dir_data_csv, help="data_csv dir")
    parser.add_argument("--data_txt", type=str, default=dir_data_txt, help="data_txt dir")
    parser.add_argument("--data_set", type=str, default=dir_data_set, help="data_set dir")
    parser.add_argument("--cut_off", type=int, default=50, help="cutoff long-tailed categorical values")
    FLAGS, unparsed = parser.parse_known_args()
    print("threads -------------- ", FLAGS.threads)
    print("csv_dir -------------- ", FLAGS.data_csv)
    print("txt_dir -------------- ", FLAGS.data_txt)
    print("set_dir -------------- ", FLAGS.data_set)
    print("cutoff --------------- ", FLAGS.cut_off)

    is_csv = 0
    if is_csv == 0:
        # 特征探索分析
        csv2txt_eda(FLAGS.data_csv)
    elif is_csv == 1:
        # CSV转TXT
        csv2txt(FLAGS.data_csv, FLAGS.data_txt)
    elif is_csv == 2:
        # 特征预处理
        preprocess(FLAGS.data_txt, FLAGS.data_set)
    else:
        # 产生最终结果
        csv2csv(FLAGS.data_txt, FLAGS.data_set)
    pass
