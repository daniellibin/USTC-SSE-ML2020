# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:55:18 2018
@author: Administrator
"""

from numpy import *
from stumpClassify import *
from ROC_plot import  *
#adaBoost算法
#@dataArr：数据矩阵
#@classLabels:标签向量
#@numIt:迭代次数    
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''
    @adaBoost算法
    @dataArr：数据矩阵
    @classLabels:标签向量
    @numIt:
    '''
    #弱分类器相关信息列表
    weakClassArr=[]
    #获取数据集行数
    m=shape(dataArr)[0]
    #初始化权重向量的每一项值相等
    D=mat(ones((m,1))/m)
    #累计估计值向量
    aggClassEst=mat((m,1))
    #循环迭代次数
    for i in range(numIt):
        #根据当前数据集，标签及权重建立最佳单层决策树
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        #打印权重向量
        print("D:",D.T)
        #求单层决策树的系数alpha
        alpha=float(0.5*log((1.0-error)/(max(error,1e-16))))
        #存储决策树的系数alpha到字典
        bestStump['alpha']=alpha
        #将该决策树存入列表
        weakClassArr.append(bestStump)
        #打印决策树的预测结果
        print("classEst:",classEst.T)
        #预测正确为exp(-alpha),预测错误为exp(alpha)
        #即增大分类错误样本的权重，减少分类正确的数据点权重
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        #更新权值向量
        D=multiply(D,exp(expon))
        D=D/D.sum()
        #累加当前单层决策树的加权预测值
        aggClassEst = aggClassEst + alpha * classEst
        #aggClassEst = array(aggClassEst)
        print("aggClassEst",aggClassEst.T)
        #求出分类错的样本个数
        aggErrors=multiply(sign(aggClassEst)!=\
                    mat(classLabels).T,ones((m,1)))
        #计算错误率
        errorRate=aggErrors.sum()/m
        print("total error:",errorRate,"\n")
        #错误率为0.0退出循环
        if errorRate==0.0:break
    #返回弱分类器的组合列表
    return weakClassArr


#测试adaBoost，adaBoost分类函数
#@datToClass:测试数据点
#@classifierArr：构建好的最终分类器
def adaClassify(datToClass,classifierArr):
    #构建数据向量或矩阵
    dataMatrix=mat(datToClass)
    #获取矩阵行数
    m=shape(dataMatrix)[0]
    #初始化最终分类器
    aggClassEst=mat(zeros((m,1)))
    #遍历分类器列表中的每一个弱分类器
    for i in range(len(classifierArr)):
        #每一个弱分类器对测试数据进行预测分类
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                classifierArr[i]['thresh'],
                                classifierArr[i]['ineq'])
        #对各个分类器的预测结果进行加权累加
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print('aggClassEst',aggClassEst)
    #通过sign函数根据结果大于或小于0预测出+1或-1
    return sign(aggClassEst)

#自适应数据加载函数
def loadDataSet(filename):
    #创建数据集矩阵，标签向量
    dataMat=[];labelMat=[]
    #获取特征数目(包括最后一类标签)
    #readline():读取文件的一行
    #readlines:读取整个文件所有行
    numFeat=len(open(filename).readline().split('\t'))
    #打开文件
    fr=open(filename)
    #遍历文本每一行
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        #数据矩阵
        dataMat.append(lineArr)
        #标签向量
        if curLine[-1] == "0.000000" or curLine[-1] == '0':
            curLine[-1] = -1.0
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#训练和测试分类器
def classify():
    #利用训练集训练分类器
    datArr,labelArr=loadDataSet('horseColicTraining.txt')
    #得到训练好的分类器
    classifierArray=adaBoostTrainDS(datArr,labelArr,100)
    #利用测试集测试分类器的分类效果
    testArr,testLabelArr=loadDataSet('horseColicTest.txt')
    prediction=adaClassify(testArr,classifierArray)
    #输出错误率

    num=shape(mat(testLabelArr))[1]
    errArr=mat(ones((num,1)))
    error=errArr[prediction!=mat(testLabelArr).T].sum()
    print("the errorRate is: %.2f" % (float(error)/float((num))))
    plotROC(prediction.T,testLabelArr)


if __name__ == '__main__':
    classify()