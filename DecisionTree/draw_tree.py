import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 画线(末端带一个点)

    draw.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction',
                      va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotMidText(cntrPt, parentPt, txtString):
    '在指定位置添加文本'

    # 中间位置坐标
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]

    draw.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(my_tree, root_pos, root_content):
    """绘制决策树"""

    # 当前树的叶子数
    numLeafs = get_num_leaves(my_tree)
    # 当前树的节点信息
    sides = list(my_tree.keys())
    firstStr = sides[0]

    # 定位第一棵子树的位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)

    # 绘制当前节点到子树节点(含子树节点)的信息
    plotMidText(cntrPt, root_pos, root_content)
    plotNode(firstStr, cntrPt, root_pos, decisionNode)

    # 获取子树信息
    secondDict = my_tree[firstStr]
    # 开始绘制子树，纵坐标-1。
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    for key in secondDict.keys():  # 遍历所有分支
        # 子树分支则递归
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        # 叶子分支则直接绘制
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    # 子树绘制完毕，纵坐标+1。
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def get_num_leaves(myTree):
    # 计算决策树的叶子数

    # 叶子数
    numLeafs = 0
    # 节点信息
    sides = list(myTree.keys())
    firstStr = sides[0]
    # 分支信息
    secondDict = myTree[firstStr]

    for key in secondDict.keys():  # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += get_num_leaves(secondDict[key])
        # 叶子分支则叶子数+1
        else:
            numLeafs += 1

    return numLeafs


def getTreeDepth(myTree):
    # 计算决策树的深度

    # 最大深度
    maxDepth = 0
    # 节点信息
    sides = list(myTree.keys())
    firstStr = sides[0]
    # 分支信息
    secondDict = myTree[firstStr]

    for key in secondDict.keys():  # 遍历所有分支
        # 子树分支则递归计算
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        # 叶子分支则叶子数+1
        else:
            thisDepth = 1

        # 更新最大深度
        if thisDepth > maxDepth: maxDepth = thisDepth

    return maxDepth


def draw(inTree):
    """显示决策树"""

    # 创建新的图像并清空 - 无横纵坐标
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    draw.ax1 = plt.subplot(111, frameon=False, **axprops)

    # 树的总宽度 高度
    plotTree.totalW = float(get_num_leaves(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))

    # 当前绘制节点的坐标
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0

    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')

    plt.show()
