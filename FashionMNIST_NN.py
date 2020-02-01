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

class_names = ['T-shirt/top, T恤/上衣', 'Trouser, 裤子', 'Pullover, 套头衫', 'Dress, 连衣裙',
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
    plt.xlabel(class_names[train_labels[i]])
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

# %% 评估模型
# verbose 输出日志信息，0不输出，1带进度条输出，2不带进度条输出
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)

# %% 样本预测
# 对样本进行一次前向传播预测
predictions = model.predict(test_images)
# 得到的结果将会是1*10的数组，表示每个标签值的可能性
predictions[0]
print(np.argmax(predictions[0]))
print(test_labels[0])


# %% 绘制图像及其预测概率直方图
# 1、绘制图片及其预测值-准确度-真实值
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{}{:2.0f}{}".format(class_names[predicted_label],
                                    100 * np.max(predictions_array),
                                    class_names[true_label],
                                    color=color))


# 2、绘制预测值直方图
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)  # 不绘制网格
    plt.xticks(range(10))  # x轴数据为0-9
    plt.yticks([])  # y轴不显示数据
    # 绘制条状直方图（直方图位置，直方图值, [宽度], 颜色）
    # 10个概率值都传入了thisplot中，一同被绘制，太小的值不可见
    # print(predictions_array)
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])  # y轴上下限，概率值的取值范围即[0,1]
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# %% 单张试验
i = 14
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# %% 组合绘制
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
