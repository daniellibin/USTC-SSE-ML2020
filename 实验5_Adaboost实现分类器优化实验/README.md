***\*实验\*******\*五\**** ***\*Adaboost实现分类器优化实验\****

 

# **一、** ***\*adaboost训练流程图\****

![img](README.assets/wps1.jpg)

 

 

 

调试结果：

![img](README.assets/wps2.jpg)

 

**二、** ***\*ROC 曲线\****

![img](README.assets/wps3.jpg) 

 

优化方法：

（1）更改基分类器的数目或类型

（2）调整基分类器的相关参数

 

 

 

 

 

 

 

 

 

 

 

 

弱分类器个数改为100时：

![img](README.assets/wps4.jpg)

![img](README.assets/wps5.jpg) 

弱分类器个数改为100且buildStump中步长由10修改至100时：

![img](README.assets/wps6.jpg)

![img](README.assets/wps7.jpg) 