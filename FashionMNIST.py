# %% 导入必要的库
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# %% 加载 Fashion MINST 数据集
# 返回两个元组
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

classnames = ['T-shirt/top, T恤/上衣', 'Trouser, 裤子', 'Pullover, 套头衫', 'Dress, 连衣裙',
              'Coat, 外套', 'Sandal, 凉鞋', 'Shirt, 衬衫', 'Sneaker, 运动鞋', 'Bag, 背包',
              'Ankle boot, 短靴']

# %% 查看数据情况
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))
train_labels

# %% 查看图片
plt.figure()  # 定义一张画布
plt.imshow(train_images[0])  # 将内容绘制到画布中
plt.colorbar()  # 绘制色条
plt.grid(False)  # 不绘制网格

plt.show()  # 显示图片

# %% 数据预处理
# 归一化到[0, 1]，训练集与测试集都需要归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# %% 检查归一化情况与标签对应情况
plt.figure(figsize=(10, 10))  # 画布大小，英寸
for i in range(25):
    plt.subplot(5, 5, i + 1)  # 绘制m*n个子图，当前绘制在第i个位置
    plt.xticks([])  # 指定x轴的刻度显示（不显示）
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(classnames[train_labels[i]])
plt.show()

# %% 建立keras模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
# %% 训练模型
model.fit(train_images, train_labels, epochs=10)
