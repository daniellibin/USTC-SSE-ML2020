def createDataSet():
    dataSet=[[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels=['no surfacing','flippers']
    return dataSet,labels

from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) #nrows
    #为所有的分类类目创建字典
    labelCounts ={}
    for featVec in dataSet:
        currentLable=featVec[-1] #取得最后一列数据
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0
        labelCounts[currentLable]+=1
    #计算香农熵
    shannonEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


#定义按照某个特征进行划分的函数splitDataSet
#输入三个变量（待划分的数据集，特征，分类值）
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value :
            reduceFeatVec=featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet #返回不含划分特征的子集

#定义按照最大信息增益划分数据的函数
def chooseBestFeatureToSplit(dataSet,method="ID3"):
    numFeature=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)#香农熵
    bestInforGain=0
    bestFeature=-1
    for i in range(numFeature):
        featList=[number[i] for number in dataSet] #得到某个特征下所有值（某列）
        uniqualVals=set(featList) #set无重复的属性特征值
        newEntropy=0
        splitInfo = 0
        for value in uniqualVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet)) #即p(t)
            newEntropy+=prob*calcShannonEnt(subDataSet)#对各子集香农熵求和
            splitInfo -= prob * log(prob, 2)
        if method == "ID3":
            infoGain=baseEntropy-newEntropy #计算信息增益
        elif method == "C4.5":
            infoGain = (baseEntropy - newEntropy) / splitInfo # 计算信息增益
        #最大信息增益
        if (infoGain>bestInforGain):
            bestInforGain=infoGain
            bestFeature=i
    return bestFeature #返回特征值


from operator import *
#投票表决代码
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items,key=itemgetter(1),reversed=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels,method="ID3"):
    classList=[example[-1] for example in dataSet]
    #类别相同，停止划分
    if classList.count(classList[-1])==len(classList):
        return classList[-1]
    #长度为1，没有其他的特征可以划分，返回出现次数最多的类别
    if len(dataSet[0]) == 1: # 训练数据只给出类别数据（没给任何属性值数据），返回出现次数最多的分类名称
        return majorityCnt(classList)
    #按照信息增益最高选取分类特征属性
    bestFeat=chooseBestFeatureToSplit(dataSet,method)#返回分类的特征序号
    bestFeatLable=labels[bestFeat] #该特征的label
    myTree={bestFeatLable:{}} #构建树的字典
    del(labels[bestFeat]) #从labels的list中删除该label
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLables=labels[:] #子集合
        #构建数据的子集合，并进行递归
        myTree[bestFeatLable][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLables)
    return myTree

