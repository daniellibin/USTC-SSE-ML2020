import numpy as np
class Perceptron(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    """
    def __init__(self,eta=0.01,n_iter=10):
        self.eta=eta;
        self.n_iter=n_iter
        pass

    def net_input(self, X):
        """
        z = W0*1+W1*X1 + ...+Wn*Xn
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]
        pass

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0 , 1, -1)
        pass

    def fit(self,X,y):
        """
        输入训练数据，培训神经元，X输入样本向量，y对应样本分类
        X：shape[n_samples,n_features]
        X:[[1,2,3],[4,5,6]]
        n_samples:2
        n_features:3
        y:[1,-1]
        """

        """
        初始化权重向量为0
        加一是因为前面算法提到的w0，也就是步调函数阈值
        """
        self.w_ =np.zeros(1 + X.shape[1])
        self.errors_=[]
        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip(X,y) = [[1,2,3 1],[4,5,6 -1]]
            """
            for xi,target in zip(X,y):
                """
                update = n * (y-y')
                target为真实值，self.predict(xi)为预测值（1 or -1）
                当两者相同，即预测正确时，差值为0，不更新
                当两者不同，即预测错误时更新。
                """
                update = self.eta * (target - self.predict(xi))

                """
                xi是一个向量
                update * xi 等价：
                {W(1) = X[1]*update, W(2) = X[2]*update, W(3) = X[3]*update}
                """
                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(update !=0.0)
                self.errors_.append(errors)
                pass
            pass
        pass