import perceptron as pp
import pandas as pd
import matplotlib as mat
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
def plot_decision_regions(x, y, classifier, resolution=0.2):
    '''
    二维数据集决策边界可视化
    :parameter
    -----------------------------
    :param self: 将鸢尾花花萼长度、花瓣长度进行可视化及分类
    :param x: list 被分类的样本
    :param y: list 样本对应的真实分类
    :param classifier: method 分类器：感知器
    :param resolution:
    :return:
    -----------------------------
    '''
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    # y 去重之后的种类
    listedColormap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1# 花萼长度最小值 -1 ，最大值 +1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1# 花瓣长度最小值 -1 ，最大值 +1
    new_x1 = np.arange(x1_min, x1_max, resolution)# 将最大值，最小值向量生成二维数组xx1,xx2
    new_x2 = np.arange(x2_min, x2_max, resolution)
    xx1, xx2 = np.meshgrid(new_x1, new_x2) # 生成网格点坐标矩阵
    '''
    xx1 = [[3.2,3.4,....7.9],[3.2,3.4,....7.9],[3.2,3.4,....7.9],[3.2,3.4,....7.9]]
    xx2 = [[0,0,....0,0]],[0.2,0.2,....0.2,0.2]],[0.4,0.4,....0.4,0.4]],[0.6,0.6,....0.6,0.6]]
    xx1.ravel = [3.2,3.4,....7.9,3.2,3.4,....7.9,3.2,3.4,....7.9,3.2,3.4,....7.9]
    xx2.reval = [0,0,....0,0,0.2,0.2,....0.2,0.2,0.4,0.4,....0.4,0.4,0.6,0.6,....0.6,0.6]
    np.array([xx1.ravel(), xx2.ravel()]).T = [[3.2,0.0],[3.2,0.2],[3.2,0.4],[3.2,0.6],[3.4,0.0],[3.4,0.2],[3.4,0.4],[3.4,0.6]......]
    '''
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) # 预测值  # np.ravel()将多维数组转换为一维数组，相当于所有数据排列组合一遍，最终predict输入为(744,2)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.4, camp=listedColormap) #contourf是来绘制等高线的
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, c1 in enumerate(np.unique(y)): # c1取值为0和1
        # x[y == c1, 0],c1类别的横坐标
        # x[y == c1, 1],c1类别的纵坐标
        plt.scatter(x=x[y == c1, 0], y=x[y == c1, 1], alpha=0.8, c=listedColormap(idx),marker=markers[idx], label=c1)

df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values# 0 到 100 行，第 5 列
y = np.where(y == "Iris-setosa", -1, 1) # 将 target 值转数字化 Iris-setosa 为 -1 ，否则值为 1
x = df.iloc[0:100, [0, 2]].values# 取出 0 到 100 行，第 1 ， 3 列的值
ppn = pp.Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
plot_decision_regions(x, y, classifier=ppn)
# 防止中文乱码
zhfont1 = mat.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.title("鸢尾花花瓣、花萼边界分割", fontproperties=zhfont1)
plt.xlabel("花瓣长度 [cm]", fontproperties=zhfont1)
plt.ylabel("花萼长度 [cm]", fontproperties=zhfont1)
plt.legend(loc="uper left")
plt.show()