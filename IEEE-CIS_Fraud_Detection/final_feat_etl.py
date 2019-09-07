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

numeric_features_etl = range(1, 14)
categorical_features = range(14, 79)
numeric_clip = [500, 900, 25, 10, 10, 100, 500, 150, 550, 250,
                550, 550, 550]


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
    infer_tran = pd.read_csv(datain_dir + "\\test_transaction.csv", index_col="TransactionID")
    infer_iden = pd.read_csv(datain_dir + "\\test_identity.csv", index_col="TransactionID")
    train = train_tran.merge(train_iden, how="left", left_index=True, right_index=True)
    infer = infer_tran.merge(infer_iden, how="left", left_index=True, right_index=True)

    order = ["isFraud",
             "TransactionAmt", "dist1", "C1", "C3", "C5", "C13", "D1", "D3", "D4", "D5",
             "D10", "D11", "D15",
             "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "P_emaildomain",
             "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
             "V1", "V2", "V3", "V4", "V6", "V7", "V8", "V10", "V12", "V14",
             "V15", "V17", "V19", "V23", "V24", "V25", "V26", "V27", "V30", "V31",
             "V35", "V37", "V39", "V43", "V44", "V46", "V47", "V53", "V55", "V56",
             "V60", "V61", "V65", "V66", "V67", "V72", "V73", "V75", "V77", "V78",
             "V82", "V86", "V87", "V88", "V95"]
    order_ = order[1:]
    train_txt = train[order]
    infer_txt = infer[order_]
    train_txt.to_csv(dataou_dir + "train_feat1.txt", sep='\t', index=False, header=0)
    infer_txt.to_csv(dataou_dir + "infer_feat1.txt", sep='\t', index=False, header=0)
    infer_idx = infer["ProductCD"]
    infer_idx.to_csv(dataou_dir + "infer_idx.csv", index=True, header=True)


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

    is_csv = 1
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
