# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:54:27 2018
@author: Administrator
"""

from numpy import *
# from load_data import *
import matplotlib.pyplot as plt
import random
import numpy as np


# 辅助函函数一
# 第一个函数的作用就是用来选择alpha对的（即寻找i，j这一对）
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


# 辅助函函数二
# 第二个函数就是为了将alpha规划到[0,C]范围内
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 最简版本SMO算法
def smoSimple(dataMat, classLabels, C, toler, maxIter):
    '''
    @dataMat    ：数据列表
    @classLabels：标签列表
    @C          ：权衡因子（增加松弛因子而在目标优化函数中引入了惩罚项）
    @toler      ：容错率
    @maxIter    ：最大迭代次数
    '''
    # 将列表形式转为矩阵或向量形式
    dataMatrix = mat(dataMat)
    labelMat = mat(classLabels).transpose()
    # 初始化b=0，获取矩阵行列
    b = 0
    m, n = shape(dataMatrix)
    # 新建一个m行1列的向量
    alphas = mat(zeros((m, 1)))
    # 迭代次数为0
    iters = 0
    while (iters < maxIter):
        # 改变的alpha对数
        alphaPairsChanged = 0
        # 遍历样本集中样本
        for i in range(m):
            # 计算支持向量机算法的预测值
            fXi = float(multiply(alphas, labelMat).T * \
                        (dataMatrix * dataMatrix[i, :].T)) + b
            # 计算预测值与实际值的误差
            Ei = fXi - float(labelMat[i])
            # 如果不满足KKT条件，即labelMat[i]*fXi<1(labelMat[i]*fXi-1<-toler)
            # and alpha<C 或者labelMat[i]*fXi>1(labelMat[i]*fXi-1>toler)and alpha>0
            if (((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                    ((labelMat[i] * Ei > toler) and (alphas[i] > 0))):
                # 随机选择第二个变量alphaj
                j = selectJrand(i, m)
                # 计算第二个变量对应数据的预测值

                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * \
                                                            dataMatrix[j, :].T)) + b
                # 计算与测试与实际值的差值
                Ej = fXj - float(labelMat[j])
                # 记录alphai和alphaj的原始值，便于后续的比较
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 如何两个alpha对应样本的标签不相同
                if (labelMat[i] != labelMat[j]):
                    # 求出相应的上下边界
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L==H");continue
                # 根据公式计算未经剪辑的alphaj
                # ------------------------------------------
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                      dataMatrix[i, :] * dataMatrix[i, :].T - \
                      dataMatrix[j, :] * dataMatrix[j, :].T
                # 如果eta>=0,跳出本次循环
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                # ------------------------------------------
                # 如果改变后的alphaj值变化不大，跳出本次循环
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving\
                enough");continue
                # 否则，计算相应的alphai值
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # 再分别计算两个alpha情况下对于的b值
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] \
                     * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                # 如果0<alphai<C,那么b=b1
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                # 否则如果0<alphai<C,那么b=b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                # 否则，alphai，alphaj=0或C
                else:
                    b = (b1 + b2) / 2.0
                # 如果走到此步，表面改变了一对alpha值
                alphaPairsChanged += 1
                print("iters: %d i:%d,paird changed %d" % (iters, i, alphaPairsChanged))
        # 最后判断是否有改变的alpha对，没有就进行下一次迭代
        if (alphaPairsChanged == 0):
            iters += 1
        # 否则，迭代次数置0，继续循环
        else:
            iters = 0
        print("iteration number: %d" % iters)
    # 返回最后的b值和alpha向量
    return b, alphas


def calcWs(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    w = np.dot((np.tile(labelMat.reshape(1, -1).T, (1, 2)) * dataMat).T, alphas)
    return w.tolist()


def showClassifer(dataMat, labelMat, alphas, w, b):
    data_plus = []
    data_minus = []
    #
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1], s=30, alpha=0.7)
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    #
    x1 = max(dataMat)[0]
    x2 = min(dataMat)[0]
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1, y2 = (-b - a1 * x1) / a2, (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])
    #
    for i, alpha in enumerate(alphas):
        if 0.6 > abs(alpha) > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')
        if 0.6 == abs(alpha):
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='yellow')
    plt.show()


def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


x, y = loadDataSet("testSet.txt")
b, alphas = smoSimple(x, y, 0.1, 0.001, 40)
w = calcWs(x, y, alphas)
showClassifer(x, y, alphas, w, b)
