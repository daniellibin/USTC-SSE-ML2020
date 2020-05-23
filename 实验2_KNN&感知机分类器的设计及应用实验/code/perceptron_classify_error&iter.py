from perceptron import Perceptron
import matplotlib.pyplot as plt
import matplotlib as mat
import pandas as pd
import numpy as np
"""
训练模型并且记录错误次数，观察错误次数的变化
"""
print(__doc__)
# 加载鸢尾花数据
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)
x = df.iloc[0:100, [0, 2]].values
"""
误差数折线图
@:param eta: 0.1 学习速率
@:param n_iter：0.1 迭代次数
"""
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(x, y)
# plot 绘制折线图
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker="o")
# 防止中文乱码
zhfont1 = mat.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.xlabel("迭代次数（n_iter）", fontproperties=zhfont1)
plt.ylabel("错误分类次数（error_number）", fontproperties=zhfont1)
plt.show()