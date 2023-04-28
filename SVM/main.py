import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.svm import LinearSVC
import os


def extract_text(mail_dir):
    """
    load data from local and extract text
    """
    files = []
    labels = []
    for dir in os.listdir(mail_dir):
        subdir = os.path.join(mail_dir, dir)
        if os.path.isdir(subdir):
            for sub_sub_dir in os.listdir(subdir):
                if sub_sub_dir == "ham":
                    for fi in os.listdir(os.path.join(subdir, "ham")):
                        files += [os.path.join(subdir, "ham", fi)]
                        labels.append(1)
                elif sub_sub_dir == "spam":
                    for fi in os.listdir(os.path.join(subdir, "spam")):
                        files += [os.path.join(subdir, "spam", fi)]
                        labels.append(0)
    test_matrix = np.ndarray((len(files)), dtype=object)
    id = 0
    for fil in files:
        with open(fil, 'r', errors="ignore") as fi:
            next(fi)
            data = fi.read().replace('\n', ' ')
            test_matrix[id] = data
            id += 1
    return test_matrix, labels


# train
train_dir = "./data/train"
train_matrix, train_labels = extract_text(train_dir)
print("train size: ", train_matrix.shape[0])

# 特征提取
count_v1 = CountVectorizer(stop_words="english", max_df=0.5, decode_error="ignore", binary=True)
counts_train = count_v1.fit_transform(train_matrix)
tf_idf_transformer = TfidfTransformer()
tfidf_train = tf_idf_transformer.fit(counts_train).transform(counts_train)

# 训练分类器
model = LinearSVC()
model.fit(tfidf_train, train_labels)

# test
test_dir = "./data/test"
test_matrix, test_labels = extract_text(test_dir)
print("test size: ", test_matrix.shape[0])

count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_, stop_words="english", max_df=0.5, decode_error="ignore", binary=True)
counts_test = count_v2.fit_transform(test_matrix)
tfidf_test = tf_idf_transformer.fit(counts_test).transform(counts_test)

result = model.predict(tfidf_test)
cm = pd.DataFrame(
    confusion_matrix(test_labels, result), index=["non-spam", "spam"], columns=["non-spam", "spam"]
)

# 混淆矩阵
print(cm)

# 精度和召回率  
print("precision score: ", precision_score(test_labels, result))
print("recall score: ", recall_score(test_labels, result))

# 绘制混淆矩阵 
cm.plot(kind='barh')  
plt.xlabel('True Label')  
plt.ylabel('Predicted Label')  
plt.title('Confusion Matrix')  
plt.show()

# 绘制直方图
cm.plot(kind='barh')  
plt.xlabel('False Positive')  
plt.ylabel('True Positive')  
plt.title('False Positive Rate')  
plt.show()
