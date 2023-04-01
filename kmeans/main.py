import os
import numpy as np
import cv2

from PIL import Image as image
from KMeans import run_kmeans


def resize(file):
    image = cv2.imread(file)
    size = 800  # 缩放后的大小
    height, width = image.shape[0], image.shape[1]
    scale = height / size # 等比例缩放
    width_size = int(width / scale)

    image_resize = cv2.resize(image, (width_size, size))
    cv2.imwrite(file + "_process.jpg", image_resize)

    return file + "_process.jpg"


def load_data(file_path):
    """
    读取文件
    :param file_path: 文件的存储位置
    :return: rgb像素矩阵
    """

    f = open(file_path, "rb")  # 以二进制的方式打开图像文件
    file_data = []
    im = image.open(f)  # 导入图片
    m, n = im.size  # 得到图片的大小
    print(m, n)
    for i in range(m):
        for j in range(n):
            tmp = []
            x, y, z = im.getpixel((i, j))
            tmp.append(x / 256.0)
            tmp.append(y / 256.0)
            tmp.append(z / 256.0)
            file_data.append(tmp)
    f.close()
    return np.mat(file_data)


def draw_new_pic(K, file):
    """
    生成分割后的图片
    :param K: 簇的个数
    :param file: 原文件名
    """

    f_center = open("center_pp")

    center = []
    for line in f_center.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            tmp.append(int(float(x) * 256))
        center.append(tuple(tmp))
    print(center)
    f_center.close()

    fp = open(file, "rb")
    im = image.open(fp)
    # 新建一个图片
    m, n = im.size
    pic_new = image.new("RGB", (m, n))

    f_sub = open("sub_pp")
    i = 0
    for line in f_sub.readlines():
        index = float((line.strip().split("\t"))[0])
        index_n = int(index)
        pic_new.putpixel((int(i / n), (i % n)), center[index_n])
        i = i + 1
    f_sub.close()

    pic_new.save("pic" + file[0] + "-k-" + str(K) + ".jpg", "JPEG")


if __name__ == "__main__":
    file_name = "5.jpg"
    k = 3

    print("---------- 0.resize    ------------")
    resize_file = resize(file_name)

    print("---------- 1.load data ------------")
    data = load_data(file_name)

    print("---------- 2.run kmeans++ ------------")
    run_kmeans(data, k)

    print("---------- 3.draw new pic ------------")
    draw_new_pic(k, file_name)

    os.remove("center_pp")
    os.remove("sub_pp")
    os.remove(resize_file)
