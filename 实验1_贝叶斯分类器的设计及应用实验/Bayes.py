import math
import numpy as np
import jieba

def loadDataSet():#数据格式
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#1 侮辱性文字 ， 0 代表正常言论
    return postingList,classVec

def createVocabList(dataSet):#创建词汇表
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document) #创建并集
    return list(vocabSet)


def setWordsToVec(vocabList,inputSet):#根据词汇表，讲句子转化为向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    return returnVec

def bagOfWord2VecMN(vocabList,inputSet):#（词袋模型）根据词汇表，讲句子转化为向量
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords);p1Num = np.ones(numWords)#计算频数初始化为1
    p0Denom = 2.0;p1Denom = 2.0                  #即拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)#注意
    p0Vect = np.log(p0Num/p0Denom)#注意
    return p0Vect,p1Vect,pAbusive#返回各类对应特征的条件概率向量
                                 #和各类的先验概率


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)#注意
    p0 = sum(vec2Classify * p0Vec) + math.log(1-pClass1)#注意

    if p1 > p0:
        return 1
    else:
        return 0


def testNBO():
    postingList, classVec = loadDataSet()  # 对数据进行加载
    myVocabList = createVocabList(postingList)  # 建立词汇表
    trainMat = []
    for doc in postingList:
        trainMat.append(bagOfWord2VecMN(myVocabList,doc))
    p0Vect,p1Vect,pAbusive = trainNB0(trainMat,classVec)

    testEntry = ["It's worthless to stop my dog eating food.",
                "Please help me to solve this problem!",
                "This dog is so stupid，But that one is so cute."]
    for test in testEntry:
        doc = bagOfWord2VecMN(myVocabList,test.lower().split()) #小写转换，并完成句子切分
        print(test,'分类结果为: ',classifyNB(doc,p0Vect,p1Vect,pAbusive))

testNBO()