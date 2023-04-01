import os
import h5py
import numpy as np
import tensorflow as tf

"""
本代码通过CNN完成了对MNIST数据集的手写体识别
"""

# 取消编译器warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
模型训练参数
"""
# 训练次数
epoch = 50
"""
学习率更新方式
值为1,表示每lr_for_epochs轮固定按lr_change_rate比例更新学习率
选择2,表示记录5次学习率大小,当前轮次loss值大于前nub次（包括本次）loss平均值时,
学习率自动降为当前学习率0.1倍,当学习率降为last_lr时，训练终止，保存模型
"""
train_method = 2  # method = 1 or 2

# train_method = 1 所需的参数
# 默认学习率
learn_rate = 0.01
# 初始学习率
init_learn_rate = 0.01
# 每10轮更新一次学习率
lr_for_epochs = 10
# 更新学习率的比例
lr_change_rate = 0.5

# train_method = 2的参数
# 截止学习率值
last_lr = 0.0001

# 设置记录nub次loss值
nub = 3
# 初始化记录nub次loss值loss2
loss2 = np.zeros((nub,))
# 保存模型地址
save_path = 'data/model'

"""
神经网络参数
"""
# 输入层神经网络节点数=28*28
width_input = 784
# 第一层神经网络节点数
width_net1 = 100
# 第二层神经网络节点数
width_net2 = 100
# 输出层神经网络节点数
width_net3 = 10

# 定义网络输入层
x = np.zeros((width_input,))
# 定义网络第一层
a1 = np.zeros((width_net1,))
# 定义网络隐藏层
a2 = np.zeros((width_net2,))
# 定义网络输出层
y = np.zeros((width_net3,))

# 初始化nub个临时保存模型的参数,以便在早停前选取最优模型
w11 = np.zeros((nub, width_input, width_net1))
b11 = np.zeros((nub, width_net1))
w21 = np.zeros((nub, width_net1, width_net2))
b21 = np.zeros((nub, width_net2))
w31 = np.zeros((nub, width_net2, width_net3))
b31 = np.zeros((nub, width_net3))

# 模型参数生成
w1 = np.random.normal(0, 2 / width_input, (width_input, width_net1))
b1 = np.random.normal(0, 2 / width_net1, (width_net1,))
w2 = np.random.normal(0, 2 / width_net1, (width_net1, width_net2))
b2 = np.random.normal(0, 2 / width_net2, (width_net2,))
w3 = np.random.normal(0, 2 / width_net2, (width_net2, width_net3))
b3 = np.random.normal(0, 2 / width_net3, (width_net3,))

# 初始化参数z（其中a=sigmoid（z））
z1 = np.dot(x, w1) + b1
z2 = np.dot(a1, w2) + b2
z3 = np.dot(a2, w3) + b3

"""
functions
"""


# 激活函数
def sigmoid(v):
    return 1 / (1 + np.exp(-v))


# 损失函数
def cross_entropy_loss(y_pre, y_guess):
    return -np.sum(y_guess * np.log(y_pre) + (1 - y_guess) * np.log(1 - y_pre))


# 反向传播
def feedforward(a, w, b):
    return sigmoid(np.dot(a, w) + b)


# 保存训练结果
def save_model(w_1, w_2, w_3, b_1, b_2, b_3):
    h5f = h5py.File(save_path, 'w')
    h5f.create_dataset('w1', data=w_1)
    h5f.create_dataset('w2', data=w_2)
    h5f.create_dataset('w3', data=w_3)
    h5f.create_dataset('b1', data=b_1)
    h5f.create_dataset('b2', data=b_2)
    h5f.create_dataset('b3', data=b_3)
    h5f.close()


"""
训练过程
"""
# 获取mnist数据集
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x, test_x = train_x / 255, test_x / 255

# 归一化
X_train, X_test = tf.cast(train_x, tf.float32), tf.cast(test_x, tf.float32)
Y_train, Y_test = tf.cast(train_y, tf.float32), tf.cast(test_y, tf.float32)

# 训练过程
for n in range(0, epoch):
    # 如果是method1, 改变学习率
    if train_method == 1:
        learn_rate = init_learn_rate * lr_change_rate ** (int(n / lr_for_epochs))  # 学习率随着学习轮数指数递减

    # 打乱样本
    r = np.random.permutation(60000)
    train_x = train_x[r, :, :]
    train_y = train_y[r]

    for i in range(0, 60000):
        x = np.array(train_x[i])
        x = x.reshape(width_input, )
        z1 = np.dot(x, w1) + b1
        a1 = feedforward(x, w1, b1)
        z2 = np.dot(a1, w2) + b2
        a2 = feedforward(a1, w2, b2)
        z3 = np.dot(a2, w3) + b3
        # y=softmax(z3)
        y = feedforward(a2, w3, b3)
        y_t = np.zeros((width_net3,))
        y_t[train_y[i]] = 1
        eta3 = (-y_t / y + (1 - y_t) / (1 - y)) * sigmoid(z3) * (1 - sigmoid(z3))  # 此为反向传播过程中中间参数，下同
        # eta3=2*(y-y_t)*sigmoid(z3)*(1-sigmoid(z3))#此为反向传播过程中中间参数，下同
        eta2 = np.dot(eta3, np.transpose(w3)) * sigmoid(z2) * (1 - sigmoid(z2))
        eta1 = np.dot(eta2, np.transpose(w2)) * sigmoid(z1) * (1 - sigmoid(z1))
        b3 = b3 - learn_rate * eta3
        b2 = b2 - learn_rate * eta2
        b1 = b1 - learn_rate * eta1
        w3 = w3 - learn_rate * np.dot(a2.reshape(width_net2, 1), eta3.reshape(1, width_net3))
        w2 = w2 - learn_rate * np.dot(a1.reshape(width_net1, 1), eta2.reshape(1, width_net2))
        w1 = w1 - learn_rate * np.dot(x.reshape(width_input, 1), eta1.reshape(1, width_net1))

    loss1 = 0
    True_num = 0

    # 加载测试集，计算loss和precision
    for i in range(0, 10000):
        x = np.array(test_x[i])
        x = x.reshape(1, width_input)
        y_t = np.zeros((width_net3,))
        y_t[test_y[i]] = 1
        a1 = feedforward(x, w1, b1)
        a2 = feedforward(a1, w2, b2)

        y = feedforward(a2, w3, b3)
        if test_y[i] == np.argmax(y, axis=1):
            True_num = True_num + 1
        loss1 = loss1 + cross_entropy_loss(y, y_t)

    precision = True_num / 10000 * 100

    # 改变学习率，利用队列方式记录连续nub次loss值
    if train_method == 2:
        # 临时存储模型
        j = range(1, nub)
        k = range(0, nub - 1)
        w11[j] = w11[k]
        b11[j] = b11[k]
        w21[j] = w21[k]
        b21[j] = b21[k]
        w31[j] = w31[k]
        w11[0] = w1
        b11[0] = b1
        w21[0] = w2
        b21[0] = b2
        w31[0] = w3
        b31[0] = b3
        loss2[j] = loss2[k]
        loss2[0] = loss1

        # 判断是否改变学习率
        if loss2[0] > np.mean(loss2) and loss2[nub - 1] > 0:
            learn_rate = learn_rate * 0.1
            if learn_rate < last_lr:
                save_model(w11[np.argmin(loss2)], w21[np.argmin(loss2)], w31[np.argmin(loss2)],
                           b11[np.argmin(loss2)], b21[np.argmin(loss2)], b31[np.argmin(loss2)])
                print("epoch:", n + 1, "lr:%.6f" % learn_rate, "loss:", loss1, 'precision:%.2f' % precision, '%')
                break

    if n % lr_for_epochs == 0:
        save_model(w1, w2, w3, b1, b2, b3)

    # 输出训练结果
    print("epoch:", n + 1, "learn rate:%.6f" % learn_rate, "loss:", loss1, 'precision:%.2f' % precision, '%')
