- [1 TF.Keras基础](#1-tfkeras基础)
  - [1.1 MNIST_NN](#11-mnist_nn)
  - [1.2 FashionMNIST_NN](#12-fashionmnist_nn)
  - [1.3 TFHub与文本分类](#13-tfhub与文本分类)
  - [1.4 BasicRegression 基础回归](#14-basicregression-基础回归)
- [2 图片](#2-图片)
  - [2.1 Image_CNN 图片的加载与预处理](#21-image_cnn-图片的加载与预处理)
    - [2.1.1 图片的加载与预处理](#211-图片的加载与预处理)
    - [2.1.2 将数据处理为batch，方便训练时取用](#212-将数据处理为batch方便训练时取用)
    - [2.1.3 模型编辑与训练](#213-模型编辑与训练)
    - [2.1.4 使用缓存提升数据集的磁盘读取性能](#214-使用缓存提升数据集的磁盘读取性能)
- [3 作业](#3-作业)
  - [3.1 性别分类](#31-性别分类)
    - [使用环境](#使用环境)
    - [使用方法](#使用方法)
    - [基本思路](#基本思路)
- [代码来源/参考](#代码来源参考)


# LearnGitAndTF2.0

## 1 TF.Keras基础

### 1.1 [MNIST_NN](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/MNIST_NN.py)

手写数字识别，使用keras.models.Sequential()构建模型

### 1.2 [FashionMNIST_NN](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/FashionMNIST_NN.py)

服装识别，使用keras.Sequential()构建模型，并对最终预测结果进行了可视化

### 1.3 [TFHub与文本分类](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/TFHub_TextClassification.py)

使用TFHub下载预训练模型实现迁移学习，并使用NN进行简单的褒贬文本分类

(working...)

### 1.4 [BasicRegression 基础回归](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/BasicRegression.py)

使用`keras.utils.get_file`从 [UCI机器学习库](https://archive.ics.uci.edu/ml/index.php) 中获取auto-mpg汽车性能数据集并使用NN进行简单的线性回归

数据进行了清理、拆分、统计、特征标签分离、归一化等操作，并使用seaborn绘图库绘制联合分布图

定义模型构建函数，统计了均方误差MSE与平均绝对误差MAE，进行了小批量训练

均方误差(MSE)是回归问题中常见的**损失函数**，平均绝对误差(MAE)是回归问题中常用的**评估指标**

在完整训练中，使用callbacks回调自定义函数显示训练进度，并使用history记录训练过程数据，如loss、mae、mse等并将训练过程数据可视化

检视训练过程，发现验证集误差在一定epoch后不降反升，使用early_stop回调函数来提前终止训练过程以达到较好效果（防止过拟合）

最后对测试集的数据进行预测，并可视化以查看拟合曲线与误差分布。

## 2 图片

### 2.1 [Image_CNN 图片的加载与预处理](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/Image_CNN.py)

#### 2.1.1 图片的加载与预处理

1. 使用`keras.utils.get_file()`下载花卉图片数据
2. 对图片数据进行一系列文件操作，包括解压--读入--显示
3. 根据文件夹确定标签--分配索引
4. 使用`tf.data.Dataset.from_tensor_slices()`对路径进行切片，得到路径字符串数据集
5. 再使用`.map()`根据路径映射图片，同时进行预处理
6. 使用相同切片方法得到标签数据集
7. 该方法的得到的预处理后的图片与标签数据集具有相同的顺序，故使用`tf.data.Dataset.zip()`打包(亦可在切片时直接打包，再预处理图片)

#### 2.1.2 将数据处理为batch，方便训练时取用

1. `.shuffle(buffer_size)`打乱数据，buffer_size缓冲区大小，保证数据被充分打乱（buffer_size=image_count）
2. `.repeat()`使数据不断重复
3. `.batch(BATCH_SIZE)`设置每个batch读取的数据量
4. `.prefetch(buffer_size)`模型训练过程中，从后台预先取出部分数据降低读盘等待（buffer_size=AUTOTUNE）
5. 可选`.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))`以融合第1、2步，以降低延迟（延迟主要由缓冲区填充造成）

#### 2.1.3 模型编辑与训练

1. `tf.keras.applications`下载模型用于迁移学习（路径为./home/.keras/models）
2. `.trainable`设置迁移网络的参数是否可训练
3. `help(keras_applications.mobilenet_v2.preprocess_input)`可查看该模型输入与输出的数据格式
4. 取出一个batch的图片，传递给模型，查看结果是否符合预期

```
image_batch, label_batch = next(iter(ds))
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)
```

5. `tf.keras.Sequential`构建迁移学习模型--试验模型--编译模型--训练模型

#### 2.1.4 使用缓存提升数据集的磁盘读取性能

1. 量化性能（将2个epoch量的数据集读入内存）
2. 在`.shuffle()`之前，使用`.cache()`对数据进行缓存，内存足够时尤为有效
3. 内存不足时，将缓存生成缓存文件`.cache(filename='./cache.tf-data')`，再次启动数据集时速度见显著提升
4. 与远程服务器传输数据集时，可采用TFRecord()将数据打包（working...）

注：

- iteration：1次迭代，每次迭代更新1次网络参数（training step）
- batch_size：每次迭代所使用的的样本量
- epoch：1个epoch即遍历1次整个样本

## 3 作业

### 3.1 [性别分类](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/predictSexual.py)

> 对”student.xls”文件进行读取身高和体重数据，并对学生的性别进行分类

#### 使用环境

- python 3.7

- tensorflow 1.13.1 （及以上）

- matplotlib.pyplot （用于绘制图表）

- pandas （用于读取excel数据并处理）

- numpy 

#### 使用方法

- 1 下载或拷贝[`predictSexual.py`](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/predictSexual.py)文件
- 2 将`dataset_path = "./dataset/student.xls"`修改为自己的excel文件路径
- 3 直接运行文件，即可看到最终效果
- 注：亦可使用`PyCharm`自带的`Scientific Mode（科学模式）`，逐块运行代码

#### 基本思路

- 1 使用`pandas`从`student.xls`读入数据
- 2 将性别转化为`one-hot`编码，并去除多余数据
- 3 拆分训练集与测试集
- 4 分离输入数据`data`（身高、体重）与预测标签`label`（性别）
- 5 借助`pandas`对数据进行归一化处理
- 6 使用`tf.keras.Sequential`构建模型
- 7 训练数据
- 8 验证模型
- 9 绘制边界图（可选）

![image](https://github.com/KPollux/LearnGitAndTF2.0/blob/master/images/decisionBoundray_87%25preSex.png)






------------------------------------------------------------------------------------------

## 代码来源/参考

[1] [怎么写md](https://www.jianshu.com/p/f378e3f2e7e1)

[2] [Github .md目录生成](https://github.com/KPollux/TOC_generator)

[3] [TensorFlow2.x 官方教程](https://tensorflow.google.cn/tutorials/)

[4] [Keras 官方文档](https://keras.io/zh/)

[5] [matplotlib.pyplot 官方文档](https://matplotlib.org/index.html)

[6] [Pandas 官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

[7] [Seaborn 官方文档](http://seaborn.pydata.org/tutorial.html)
