import os
import torchvision.datasets.mnist as mnist
from skimage import io

"""
本代码实现了数据集可视化，由于生成的图片较多，没有在目录下保留运行结果，可以手动运行并在./data目录下查看结果
"""

root = "./data"
train_set = (
    mnist.read_image_file(os.path.join('./data', 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join('./data', 'train-labels-idx1-ubyte'))
)
test_set = (
    mnist.read_image_file(os.path.join('./data', 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join('./data', 't10k-labels-idx1-ubyte'))
)
print("training set :", train_set[0].size())
print("test set :", test_set[0].size())


def convert_to_img(select):
    f = open(root + 'train.txt', 'w')
    if select == "train":
        data_path = root + '/train/'
    else:
        data_path = root + '/test/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for i, (img, label) in enumerate(zip(train_set[0], train_set[1])):
        img_path = data_path + str(i) + '.jpg'
        io.imsave(img_path, img.numpy())
        f.write(img_path + ' ' + str(label) + '\n')
    f.close()


convert_to_img("train")  # 转换训练集
convert_to_img("test")  # 转换测试集
