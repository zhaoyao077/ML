import pandas as pd
import pickle

from math import log2

import draw_tree


def drop_columns(csv_file):
    """数据预处理,去除冗余列"""
    df = pd.read_csv(csv_file)
    df = df.drop(['recordId', 'drugName', 'condition', 'reviewComment', 'date'], axis=1)
    new_file = csv_file.replace(".csv", "_process.csv")
    df.to_csv(new_file, index=False, encoding="utf-8")
    return new_file


def drop_dirty_lines(file):
    """数据预处理，去除脏数据"""
    # 读取文件
    with open(file, 'r') as f:
        lines = f.readlines()

    # 删去字符","出现次数不等于2的行
    new_lines = []
    for line in lines:
        if line.count(',') == 2:
            new_lines.append(line)

    # 将结果覆盖原文件
    with open(file, 'w') as f:
        f.writelines(new_lines)


def calc_entropy(data):
    """
    计算数据集的熵
    """
    labels = data.iloc[:, -1]
    label_counts = labels.value_counts()
    entropy = 0
    for count in label_counts:
        p = count / len(labels)
        entropy -= p * log2(p)
    return entropy


def calc_info_gain(data, feature):
    """
    计算信息增益
    """
    entropy_before = calc_entropy(data)
    entropy_after = 0
    for value in data[feature].unique():
        sub_data = data[data[feature] == value]
        sub_entropy = calc_entropy(sub_data)
        entropy_after += len(sub_data) / len(data) * sub_entropy
    return entropy_before - entropy_after


def id3(data):
    """
    ID3算法实现
    """
    # 如果数据集中所有样本属于同一类别，则返回该类别
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[0, -1]

    # 如果数据集中没有特征，则返回出现次数最多的类别
    if data.shape[1] == 1:
        return data.iloc[:, -1].value_counts().idxmax()

    # 计算每个特征的信息增益，并选择信息增益最大的特征作为当前节点的划分标准
    info_gains = {}
    for feature in data.columns[:-1]:
        info_gains[feature] = calc_info_gain(data, feature)
    best_feature = max(info_gains, key=info_gains.get)

    # 根据所选特征的不同取值建立子节点
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        sub_data = data[data[best_feature] == value].drop(best_feature, axis=1)
        tree[best_feature][value] = id3(sub_data)

    return tree


if __name__ == '__main__':

    file_name = 'training.csv'

    # 数据预处理,去除冗余列
    new_file = drop_columns(file_name)

    # 数据预处理，去除脏数据
    drop_dirty_lines(new_file)

    # 读取csv文件中的数据
    data = pd.read_csv(new_file)

    # 使用ID3算法构建决策树模型
    tree = id3(data)

    # 打印决策树模型
    print(tree)

    # 保存决策树模型到文件
    with open('tree.pickle', 'wb') as f:
        pickle.dump(tree, f)

    # 模型可视化
    draw_tree.draw(tree)
