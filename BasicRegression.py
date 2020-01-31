# %% 导包
from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# %% 下载数据集
#  Auto MPG 数据集，构建了一个用来预测70年代末到80年代初汽车燃油效率的模型。
#  为了做到这一点，我们将为该模型提供许多那个时期的汽车描述。
#  这个描述包含：气缸数，排量，马力以及重量。
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                                    cache_subdir="D:\\HomePC\\dataset")
dataset_path

# %% 使用pandas导入数据集
# [英里/加仑，汽缸，排量，马力，重量，加速度，车型年，国家分类]
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
# 使用pandas读取csv数据（类似excel），（路径，names=列名，
#                                   na_values=文件中空数据的显示方式，
#                                   comment=注释符，从发现该字符开始，后续忽略，
#                                   sep=指定文件中使用的分隔符，skipinitialspace=忽略空格）
# 20.0   6   198.0      95.00      3102.      16.5   74  1	"plymouth duster"
# 21.0   6   200.0      ?          2875.      17.0   74  1	"ford maverick"
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# %% 数据预处理1
dataset.isna().sum()  # 统计为空的值的个数
dataset = dataset.dropna()  # 删除空数据行
origin = dataset.pop('Origin')  # 删除Origin列，单独处理
dataset['USA'] = (origin == 1) * 1.0  # 新增列USA，其值为origin的类别的one-hot编码
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
# %%
dataset.tail()
# %%
# 从数据中取80%作为训练集，并打乱顺序
train_dataset = dataset.sample(frac=0.8, random_state=0)
# 从原数据中将作为训练集的部分取出，剩下的20%作为测试集
test_dataset = dataset.drop(train_dataset.index)

# %% 数据检查
# 使用seaborn绘制联合分布
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()
# %% 查看总体数据统计
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats
# %% 从标签中分离想要预测的特征
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# %% 数据预处理2：归一化
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# %% 构建模型
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary()

# %% 小批量试验模型
example_batch = normed_test_data[:10]  # 取10条数据做小规模试验
example_result = model.predict(example_batch)
example_result  # 得到预期的形状和结果类型


# %% 正式训练
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')

EPOCH = 1000
# 使用history对象存储训练进度
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCH, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()]
)
# %% 查看训练过程数据
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

# %% 训练过程数据可视化