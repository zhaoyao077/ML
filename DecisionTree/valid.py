import pandas as pd
from sklearn.metrics import f1_score

# 读取csv文件的最后一列
y_true = pd.read_csv('validation.csv').iloc[:, -1]
y_pred = pd.read_csv('validation_process.csv').iloc[:, -1]

# 计算Micro-F1分数
f1 = f1_score(y_true, y_pred, average='micro')
f2 = f1_score(y_true, y_pred, average='macro')


# 打印结果
print(f'Micro-F1: {f1}')
print(f'Macro-F1: {f2}')
