import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mat
import numpy as np

file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(file,header=None)
print(df.head(100))
# 0 到 100 行，第 5 列
y = df.iloc[0:100, 4].values
# 将 target 值转数字化 Iris-setosa 为 -1 ，否则值为 1
y = np.where(y == "Iris-setosa", -1, 1)
# 取出 0 到 100 行，第 1 ,第3列的值
x = df.iloc[0:100, [0, 2]].values
""" 鸢尾花散点图 """
# scatter 绘制点图
plt.scatter(x[0:50, 0], x[0:50, 1], color="red", marker="o", label="setosa")
plt.scatter(x[50:100, 0], x[50:100, 1], color="blue", marker="x", label="versicolor")
# 防止中文乱码
zhfont1 = mat.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
mat.font_manager.FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
plt.title("鸢尾花散点图", fontproperties=zhfont1)
plt.xlabel(u"花瓣长度", fontproperties=zhfont1)
plt.ylabel(u"萼片长度", fontproperties=zhfont1)
plt.legend(loc="upper left")
plt.show()