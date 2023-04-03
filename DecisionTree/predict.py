import pickle

import pandas as pd

from decision_tree import drop_columns, drop_dirty_lines


def predict(tree, data):
    """
    使用决策树模型对数据进行预测
    """
    feature = list(tree.keys())[0]
    sub_tree = tree[feature]
    value = data[feature].iloc[0]

    # 处理未知特征值
    if value not in sub_tree:
        return -1

    if isinstance(sub_tree[value], dict):
        return predict(sub_tree[value], data)
    else:
        return sub_tree[value]


def do_predict():
    """
    预处理文件并调用预测方法
    """
    # 从文件中加载决策树模型
    with open('tree.pickle', 'rb') as f:
        tree = pickle.load(f)

    t = type(tree)

    test_file = 'validation.csv'

    new_test_file = drop_columns(test_file)

    drop_dirty_lines(new_test_file)

    # 读取测试集数据
    test_data = pd.read_csv(new_test_file, usecols=[0, 1])

    # 使用决策树模型对测试集进行预测
    predictions = []
    for i in range(len(test_data)):
        data = test_data.iloc[[i]]
        prediction = predict(tree, data)
        predictions.append(prediction)
        test_data.loc[i, 'rating'] = prediction

    test_data.to_csv(new_test_file, index=False)


do_predict()
