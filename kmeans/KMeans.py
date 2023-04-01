import numpy as np
from random import random

FLOAT_MAX = 1e100  # 设置一个较大的值作为初始化的最小的距离


def run_kmeans(data, k):
    """
    KMeans算法的驱动方法
    :param data: 样本
    :param k: 聚簇个数
    """

    # 1、KMeans的聚簇中心初始化方法
    print("\t---------- 1.K-Means generate centers ------------")
    centroids = get_centroids(data, k)
    # 2、聚簇计算
    print("\t---------- 2.kmeans ------------")
    subCenter = do_kmeans(data, k, centroids)
    # 3、保存所属的类别文件
    print("\t---------- 3.save subCenter ------------")
    save_result("sub_pp", subCenter)
    # 4、保存聚簇中心
    print("\t---------- 4.save centroids ------------")
    save_result("center_pp", centroids)


def get_centroids(points, k):
    """
    初始化聚类中心
    :param points: 样本点
    :param k: 聚簇中心的个数
    :return: 聚簇中心
    """

    m, n = np.shape(points)
    cluster_centers = np.mat(np.zeros((k, n)))

    # 随机选择一个样本作为聚类中心
    index = np.random.randint(0, m)
    cluster_centers[0,] = np.copy(points[index,])

    # 初始化一个距离的序列
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0

        for j in range(m):
            # 对每一个样本找到最近的聚类中心点
            d[j] = nearest(points[j,], cluster_centers[0:i, ])
            # 将所有的最短距离相加
            sum_all += d[j]

        # 取得sum_all之间的随机值
        sum_all *= random()

        # 获得距离最远的样本点作为聚类中心点
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[i] = np.copy(points[j,])
            break

    return cluster_centers


def do_kmeans(data, k, centroids):
    """
    根据欧式距离不断更新聚簇中心
    :param data: 样本
    :param k: 聚簇个数
    :param centroids: 聚簇中心
    :return: 聚簇
    """

    m, n = np.shape(data)  # m:样本个数, n:维度
    subCenter = np.mat(np.zeros((m, 2)))  # 初始化每一个样本所属的类别

    change = True  # 判断是否需要重新计算聚类中心
    while change:
        change = False  # 重置

        for i in range(m):
            minDist = np.inf  # 设置样本与聚类中心之间的最小的距离，初始值为争取穷
            minIndex = 0  # 所属的类别

            for j in range(k):
                # 计算i和每个聚类中心之间的距离
                dist = distance(data[i,], centroids[j,])
                if dist < minDist:
                    minDist = dist
                    minIndex = j

            # 判断是否需要改变
            if subCenter[i, 0] != minIndex:  # 需要改变
                change = True
                subCenter[i,] = np.mat([minIndex, minDist])

        # 重新计算聚类中心
        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0  # 每个类别中的样本的个数

            for i in range(m):
                if subCenter[i, 0] == j:  # 计算第j个类别
                    sum_all += data[i,]
                r += 1

            for z in range(n):
                centroids[j, z] = sum_all[0, z] / r
                # print(r)

    return subCenter


def distance(vector_a, vector_b):
    """
    计算两个向量之间的欧式距离的平方
    :param vector_a: 向量a
    :param vector_b: 向量b
    :return: 欧式距离的平方
    """

    dist = (vector_a - vector_b) * (vector_a - vector_b).T

    return dist[0, 0]


def nearest(point, cluster_centers):
    """
    计算样本点和聚簇中心之间的最小距离
    :param point: 样本点
    :param cluster_centers: 聚簇中心
    :return: 最小距离
    """

    min_dist = FLOAT_MAX
    m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
    for i in range(m):
        # 计算point与每个聚类中心之间的距离
        d = distance(point, cluster_centers[i,])
        # 选择最短距离
        if min_dist > d:
            min_dist = d
    return min_dist


def save_result(file_name, source):
    """
    保存分类数据到文件中
    :param file_name: 文件名
    :param source: 数据
    :return:
    """

    m, n = np.shape(source)  # m:样本个数, n:维度
    f = open(file_name, "w")

    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(str(source[i, j]))
        f.write("\t".join(tmp) + "\n")

    f.close()
