"""
使用keras实现NN网络进行电影文本分类
同时利用TF.Hub实现迁移学习
训练数据IMDB来自TensorFlow数据集
"""

# %% 导入必要的包
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

# !pip install -q tensorflow-hub
# !pip install -q tfds-nightly
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# %% 下载IDMB数据集
# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到 15,000
# 个训练样本, 10,000 个验证样本以及 25,000 个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True,
    data_dir="D:\\HomePC\\dataset\\tensorflow_datasets\\downloads\\aclImdb\\aclImdb\\",
    download=False)

# %% 探索数据
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
# 打印前10个样本与标签
train_examples_batch
train_labels_batch

# %% 构建预训练模型
# 如何表示文本、模型有多少层、有多少隐藏单元
# 文本表示：将句子转化为嵌入向量（embeddings vectors）
# 无需担心文本预处理，可以使用迁移学习，长度固定
# 使用迁移学习（与训练模型）对句子进行转化
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

# %% 构建完整模型
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %% 训练模型
history = model.fit(train_data.shuffle(10000).batch(512),
                    epoch=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# %% 评估模型
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
