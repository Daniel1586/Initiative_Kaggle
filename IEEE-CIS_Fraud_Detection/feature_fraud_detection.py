#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Preprocess ieee-fraud-detection dataset.
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
import sys
import random
import argparse
import collections
import pandas as pd

# There are 371 numeric features and 49 categorical features
# 数值特征I1-I371(整数), 离散特征C1-C49
numeric_features = range(1, 372)
categorical_features = range(372, 421)

# Clip numeric features. The clip point for each numeric feature is
# derived from the 95% quantile of the total values in each feature
# 数值特征的阈值,若数值特征超过阈值,则该特征值置为阈值[剔除异常值]
# 阈值来源--https://github.com/PaddlePaddle/models/blob/develop/legacy/deep_fm/preprocess.py
numeric_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


def csv2txt(datain_dir, dataou_dir):
    # import data [index_col指定哪一列数据作为行索引,返回DataFrame]
    train_tran = pd.read_csv(datain_dir + "\\train_tran.csv", index_col="TransactionID")
    train_iden = pd.read_csv(datain_dir + "\\train_iden.csv", index_col="TransactionID")
    train = train_tran.merge(train_iden, how="left", left_index=True, right_index=True)
    order = ["isFraud", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
             "addr1", "addr2", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4",
             "M5", "M6", "M7", "M8", "M9",
             "TransactionDT", "TransactionAmt", "dist1", "dist2", "C1", "C2", "C3", "C4",
             "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "D1", "D2",
             "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14",
             "D15", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11",
             "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22",
             "V23", "V24", "V25", "V26", "V27", "V28", "V29", "V30", "V31", "V32", "V33",
             "V34", "V35", "V36", "V37", "V38", "V39", "V40", "V41", "V42", "V43", "V44",
             "V45", "V46", "V47", "V48", "V49", "V50", "V51", "V52", "V53", "V54", "V55",
             "V56", "V57", "V58", "V59", "V60", "V61", "V62", "V63", "V64", "V65", "V66",
             "V67", "V68", "V69", "V70", "V71", "V72", "V73", "V74", "V75", "V76", "V77",
             "V78", "V79", "V80", "V81", "V82", "V83", "V84", "V85", "V86", "V87", "V88",
             "V89", "V90", "V91", "V92", "V93", "V94", "V95", "V96", "V97", "V98", "V99",
             "V100", "V101", "V102", "V103", "V104", "V105", "V106", "V107", "V108", "V109",
             "V110", "V111", "V112", "V113", "V114", "V115", "V116", "V117", "V118", "V119",
             "V120", "V121", "V122", "V123", "V124", "V125", "V126", "V127", "V128", "V129",
             "V130", "V131", "V132", "V133", "V134", "V135", "V136", "V137", "V138", "V139",
             ]
    train.to_csv(dataou_dir + "train.txt", sep='\t', index=False)

    #  V140, V141, V142, V143, V144, V145, V146, V147, V148, V149, V150, V151,
    # V152, V153, V154, V155, V156, V157, V158, V159, V160, V161, V162, V163, V164, V165,
    # V166, V167, V168, V169, V170, V171, V172, V173, V174, V175, V176, V177, V178, V179,
    # V180, V181, V182, V183, V184, V185, V186, V187, V188, V189, V190, V191, V192, V193,
    # V194, V195, V196, V197, V198, V199, V200, V201, V202, V203, V204, V205, V206, V207,
    # V208, V209, V210, V211, V212, V213, V214, V215, V216, V217, V218, V219, V220, V221,
    # V222, V223, V224, V225, V226, V227, V228, V229, V230, V231, V232, V233, V234, V235,
    # V236, V237, V238, V239, V240, V241, V242, V243, V244, V245, V246, V247, V248, V249,
    # V250, V251, V252, V253, V254, V255, V256, V257, V258, V259, V260, V261, V262, V263,
    # V264, V265, V266, V267, V268, V269, V270, V271, V272, V273, V274, V275, V276, V277,
    # V278, V279, V280, V281, V282, V283, V284, V285, V286, V287, V288, V289, V290, V291,
    # V292, V293, V294, V295, V296, V297, V298, V299, V300, V301, V302, V303, V304, V305,
    # V306, V307, V308, V309, V310, V311, V312, V313, V314, V315, V316, V317, V318, V319,
    # V320, V321, V322, V323, V324, V325, V326, V327, V328, V329, V330, V331, V332, V333,
    # V334, V335, V336, V337, V338, V339
    # id_01, id_02, id_03, id_04, id_05, id_06, id_07, id_08, id_09, id_10, id_11, id_12,
    # id_13, id_14, id_15, id_16, id_17, id_18, id_19, id_20, id_21, id_22, id_23, id_24,
    # id_25, id_26, id_27, id_28, id_29, id_30, id_31, id_32, id_33, id_34, id_35, id_36,
    # id_37, id_38, DeviceType, DeviceInfo
    pass


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
                        val = int(val)
                        if val > numeric_clip[i]:
                            val = numeric_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


def preprocess(datain_dir, dataou_dir):

    print("========== 1.Preprocess categorical and numeric features...")
    c_feat = CategoryDictGenerator(len(categorical_features))
    c_feat.build(datain_dir + "train.txt", categorical_features, cutoff=FLAGS.cut_off)
    n_feat = NumericFeatureGenerator(len(numeric_features))
    n_feat.build(datain_dir + "train.txt", numeric_features)

    print("========== 2.Generate index of feature embedding ...")
    # 生成数值特征编号: I1-I13
    output = open(dataou_dir + "embed.set", 'w')
    for i in numeric_features:
        output.write("{0} {1}\n".format('I'+str(i), i))

    dict_sizes = list(c_feat.dicts_sizes())
    c_feat_offset = [n_feat.num_feature]
    # 生成离散特征编号: C1|xxxx XX (不同离散特征第一个特征编号的特征统一为<unk>)
    for i in range(1, len(categorical_features)+1):
        offset = c_feat_offset[i - 1] + dict_sizes[i - 1]
        c_feat_offset.append(offset)
        for key, val in c_feat.dicts[i-1].items():
            output.write("{0} {1}\n".format('C'+str(i)+'|'+key, c_feat_offset[i - 1]+val+1))

    random.seed(0)
    # 90% data are used for training, and 10% data are used for validation
    print("========== 3.Generate train/valid/test dataset ...")
    with open(dataou_dir + "train.set", 'w') as out_train:
        with open(dataou_dir + "valid.set", 'w') as out_valid:
            with open(datain_dir + "train.txt", 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_val = []
                    # numeric features normalized to [0,1]
                    for i in range(0, len(numeric_features)):
                        val = n_feat.gen(i, features[numeric_features[i]])
                        feat_val.append(str(numeric_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                    # categorical features one-hot embedding
                    for i in range(0, len(categorical_features)):
                        val = c_feat.gen(i, features[categorical_features[i]]) + c_feat_offset[i] + 1
                        feat_val.append(str(val) + ':1')

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_val)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_val)))

    with open(dataou_dir + "tests.set", 'w') as out_test:
        with open(datain_dir + "train_test.txt", 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_val = []
                # numeric features normalized to [0,1]
                for i in range(0, len(numeric_features)):
                    val = n_feat.gen(i, features[numeric_features[i]])
                    feat_val.append(str(numeric_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

                # categorical features one-hot embedding
                for i in range(0, len(categorical_features)):
                    val = c_feat.gen(i, features[categorical_features[i]]) + c_feat_offset[i] + 1
                    feat_val.append(str(val) + ':1')

                label = features[0]
                out_test.write("{0} {1}\n".format(label, ' '.join(feat_val)))

    print("========== 4.Generate infer dataset ...")
    with open(dataou_dir + "infer.set", 'w') as out_infer:
        with open(datain_dir + "test.txt", 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                feat_val = []
                for i in range(0, len(numeric_features)):
                    val = n_feat.gen(i, features[numeric_features[i] - 1])
                    feat_val.append(str(numeric_features[i]) + ':' + "{0:.6f}".format(val).rstrip('0').rstrip('.'))

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
    parser.add_argument("--cut_off", type=int, default=200, help="cutoff long-tailed categorical values")
    FLAGS, unparsed = parser.parse_known_args()
    print("threads -------------- ", FLAGS.threads)
    print("csv_dir -------------- ", FLAGS.data_csv)
    print("txt_dir -------------- ", FLAGS.data_txt)
    print("set_dir -------------- ", FLAGS.data_set)
    print("cutoff --------------- ", FLAGS.cut_off)

    # CSV转TXT
    csv2txt(FLAGS.data_csv, FLAGS.data_txt)
    # 特征预处理
    # preprocess(FLAGS.data_txt, FLAGS.data_set)
    pass
