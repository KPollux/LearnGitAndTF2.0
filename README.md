# LearnGitAndTF2.0
## 1 TF.Keras基础
### 1.1 [MNIST_NN](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/MNIST_NN.py)
手写数字识别，使用keras.models.Sequential()构建模型
### 1.2 [FashionMNIST_NN](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/FashionMNIST_NN.py)
服装识别，使用keras.Sequential()构建模型，并对最终预测结果进行了可视化
### 1.3 [TFHub与文本分类](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/TFHub_TextClassification.py)
使用TFHub下载预训练模型实现迁移学习，并使用NN进行简单的褒贬文本分类
(working...)
### 1.4 [基础回归](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/BasicRegression.py)
使用keras从[UCI机器学习库](https://archive.ics.uci.edu/ml/index.php)中获取auto-mpg汽车性能数据集并使用NN进行简单的线性回归
数据进行了清理、拆分、统计、特征标签分离、归一化等操作，并使用seaborn绘图库绘制联合分布图
定义模型构建函数，统计了均方误差MSE与平均绝对误差MAE，进行了小批量训练
在完整训练中，使用callbacks回调自定义函数显示训练进度，并使用history记录训练过程数据，如loss、mae、mse等并将训练过程数据可视化
(working...)

## 代码来源/参考
[1] [怎么写md](https://www.jianshu.com/p/f378e3f2e7e1)

[2] [TensorFlow2.x 官方教程](https://tensorflow.google.cn/tutorials/)

[3] [Keras 官方文档](https://keras.io/zh/)

[4] [matplotlib.pyplot 官方文档](https://matplotlib.org/index.html)

[5] [Pandas 官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

[6] [Seaborn 官方文档](http://seaborn.pydata.org/tutorial.html)
