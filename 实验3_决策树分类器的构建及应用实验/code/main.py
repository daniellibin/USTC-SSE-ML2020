import tree
import treeplotter
import pandas as pd
import numpy as np

'''
lenses,lensesLabels = tree.createDataSet()
lensesTree = tree.createTree(lenses,lensesLabels)
print(lensesTree)
treeplotter.createPlot(lensesTree)
'''

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
x = df.iloc[:,[0,1,2,3]].values
y = df.iloc[:,4].values
labels=["sepal length<5. 55","setal width>3. 35","petal length<2. 45","petal width>0. 8"]
mydat = list(np.where(x[:,0]<5.5, 1, 0))
mydat = np.vstack([mydat,list(np.where(x[:,1]<3.3, 1, 0))])
mydat = np.vstack([mydat,list(np.where(x[:,2]<2., 1, 0))])
mydat = np.vstack([mydat,list(np.where(x[:,3]<1, 1, 0))])
mydat = mydat.transpose(1,0)
mydat = np.column_stack((mydat,y))
mydat = list(mydat)
for i in range(len(mydat)):
    mydat[i] = list(mydat[i])

lensesTree = tree.createTree(mydat,labels,"C4.5")
print(lensesTree)
treeplotter.createPlot(lensesTree)

