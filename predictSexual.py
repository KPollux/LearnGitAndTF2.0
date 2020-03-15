import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %% 读取数据
dataset_path = "./dataset/student.xls"

print(dataset_path)
column_names = ['No', 'Sexual', 'Height', 'Weight']
raw_dataset = pd.read_excel(dataset_path, names=column_names)  # 重新指定英文列名

dataset = raw_dataset.copy()  # 复制获取到的数据
print(dataset.tail(3))  # 输出最后3行值
'''
      No Sexual  Height  Weight
257  258      女   162.1    51.3
258  259      女   165.9    58.1
259  260      男   168.7    67.8
'''
# %% 将性别转化为one-hot编码
sexual = dataset.pop('Sexual')  # 取出性别列
dataset['Male'] = (sexual == '男') * 1.0  # 新增男性列，值0表示为女性
# dataset['Female'] = (sexual == '女') * 1.0
print(dataset.tail(3))  # 输出最后3行值
'''
      No  Height  Weight  Male
257  258   162.1    51.3   0.0
258  259   165.9    58.1   0.0
259  260   168.7    67.8   1.0
'''
# %% 去掉多余的No列
dataset.pop('No')
# %% 检查数据分布
plt.figure()
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(dataset.loc[dataset['Male'] == 1]['Height'],
            dataset.loc[dataset['Male'] == 1]['Weight'], c='blue')
plt.scatter(dataset.loc[dataset['Male'] == 0]['Height'],
            dataset.loc[dataset['Male'] == 0]['Weight'], c='red')
plt.legend(['male', 'female'])
plt.show()
# %% 由于训练集太少，取90%训练集，10%测试集
train_dataset = dataset.sample(frac=0.9, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
'''
len(dataset)  # 260
print(len(train_dataset))  # 234
len(test_dataset)  # 26
'''
# %% 数据检查：查看总体数据统计
train_stats = train_dataset.describe()  # 包含每列的计数、均值、标准差、最小值、最大值等信息
train_stats.pop("Male")
train_stats = train_stats.transpose()
# %%
print(train_stats)
'''
        count        mean        std    min      25%     50%      75%    max
Height  234.0  170.809402   7.430294  151.7  165.725  171.00  175.800  188.0
Weight  234.0   65.482479  12.659946   39.3   55.700   62.95   74.225  107.2
'''
# %% 从标签中分离需要预测的特征
train_labels = train_dataset.pop('Male')
test_labels = test_dataset.pop('Male')
# %%
print(train_labels.tail(3))  # 数据带有序号
'''
77     1.0
25     1.0
165    0.0
'''


# %% 数据预处理：归一化
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print(normed_train_data.tail(3))
'''
       Height    Weight
77  -0.216600 -0.598935
25   1.008116 -0.124999
165 -2.006570 -1.057072
'''
# %% 查看归一化后的数据
plt.figure()
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(normed_train_data.loc[train_labels == 1]['Height'],
            normed_train_data.loc[train_labels == 1]['Weight'], c='blue')
plt.scatter(normed_train_data.loc[train_labels == 0]['Height'],
            normed_train_data.loc[train_labels == 0]['Weight'], c='red')
plt.legend(['male', 'female'])
plt.show()
# %% 构建模型
model = tf.keras.Sequential([
    # 第一层全连接层，64个隐藏单元，激活函数为relu，输入维度为train_dataset的数据个数（==2）
    tf.keras.layers.Dense(8, activation=tf.nn.softmax, input_shape=[len(train_dataset.keys())]),
    # 第二层相似，二元分类时考虑使用sigmoid激活函数
    # tf.keras.layers.Dense(4),
    # 输出层
    tf.keras.layers.Dense(1)
])

# optimizer = tf.keras.optimizers.RMSprop(0.001)  # 使用RMSProp有助更快收敛，训练步长初始化为0.001
optimizer = tf.keras.optimizers.SGD(0.05)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()
# %% 小批量试验模型
# 模型权重已自动随机初始化
example_batch = normed_test_data[:10]  # 取10条数据做小规模试验
example_result = model.predict(example_batch)
example_result  # 得到预期的形状和结果类型


# %% 正式训练
class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


# 防止过拟合，如果10轮都没有改进，则停止
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# 训练1000轮
EPOCH = 1000

# 从训练集再分20%作为验证集，由于轮次较多，verbose日志混乱，故打点显示训练进度
# 由于数据量不大，故进行全批量训练
# 开始训练，训练日志将保存在history中
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCH, validation_split=0.2, verbose=1,
    callbacks=[early_stop, PrintDot()]
)
# %% 查看训练过程数据
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())
print(hist.describe())


# %% 训练过程数据可视化
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # 绘制训练集与测试集的平均绝对误差图像
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [Sexual]')
    plt.plot(hist['epoch'], hist['accuracy'],
             label='Train Accuracy')  # 以批次为横坐标，mae为纵坐标，并制定图例名称
    plt.plot(hist['epoch'], hist['val_accuracy'],
             label='Val Accuracy')
    # plt.ylim([0, 5])
    plt.legend()  # 加上图例
    plt.show()


plot_history(history)

# %% 验证模型
loss, acc = model.evaluate(normed_test_data, test_labels, verbose=1)

print("Testing set acc: {:5.2f} Sexual".format(acc))  # Testing set acc:  0.92 Sexual


# %% TODO() 绘制决策边界
def produce_random_data(u_x, d_x, u_y, d_y, num):
    X1 = np.random.uniform(u_x, d_x, num)
    X2 = np.random.uniform(u_y, d_y, num)
    X = np.vstack((X1, X2))
    return X.transpose()


def collect_boundary_data(v_xs):
    global prediction
    X = np.empty([1, 2])
    X = list()
    for i in range(len(v_xs)):
        x_input = v_xs[i]
        x_input.shape = [1, 2]
        # y_pre = sess.run(prediction, feed_dict={xs: x_input})
        y_pre = model.predict(x_input)
        if abs(y_pre - 0) < 0.5:
            X.append(v_xs[i])
    return np.array(X)


# 产生空间随机数据
X_NUM = produce_random_data(-3, 3, -3, 4, 5000)
# 边界数据采样
# X_b = collect_boundary_data(X_NUM)
# # 画出数据
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# # 设置坐标轴名称
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.xlabel('Height')
# plt.ylabel('Weight')
# plt.scatter(normed_train_data.loc[train_labels == 1]['Height'],
#             normed_train_data.loc[train_labels == 1]['Weight'], c='blue')
# plt.scatter(normed_train_data.loc[train_labels == 0]['Height'],
#             normed_train_data.loc[train_labels == 0]['Weight'], c='red')
# plt.legend(['male', 'female'])
# # 用采样的边界数据拟合边界曲线 7次曲线最佳
# z1 = np.polyfit(X_b[:, 0], X_b[:, 1], 7)
# p1 = np.poly1d(z1)
# x = X_b[:, 0]
# x.sort()
# yvals = p1(x)
# plt.plot(x, yvals, 'r', label='boundray line')
# plt.legend(loc=4)
# # plt.ion()
# plt.show()
