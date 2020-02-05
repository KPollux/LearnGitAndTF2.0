"""
1 下载数据，获取数据路径
2 显示图片，检查数据情况
3 为分好文件夹的数据按打好标签
"""
# %% 导包并下载所需数据
import tensorflow as tf

# %%
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 可以让程序自动的选择最优的线程并行个数

# %% 下载数据（亦可自行下载，只需使data_root正确获取路径即可）
import pathlib

data_root_orig = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    fname='flower_photos', untar=True,
    cache_subdir="D:\\HomePC\\dataset")
data_root = pathlib.Path(data_root_orig)
print(data_root)

# %% 检查下载目录是否正常解压
# 相当于ls/dir命令，遍历该目录下所有文件
for item in data_root.iterdir():
    print(item)

# %% 将图片路径读入并打乱顺序
import random

all_image_paths = list(data_root.glob('*/*'))  # 将目录下所有文件路径都存入了list中，格式为WindowsPath('Path')
all_image_paths = [str(path) for path in all_image_paths]  # 将WindowsPath中的Path单独取出
random.shuffle(all_image_paths)  # 打乱顺序
# image_count = len(all_image_paths)
# image_count
# all_image_paths[:10]

# %% 检查图片--处理图片路径
import os

attributions = (data_root / "LICENSE.txt").open(encoding='utf-8').readlines()[4:]  # 将LICENSE中的文字从第4行开始读入
attributions = [line.split(' CC-BY') for line in attributions]  # 将内容以' CC-BY'分割，两两一组放入元组中
attributions = dict(attributions)  # 将元组转化为字典
attributions

# %% 检查图片--从路径显示图片
import IPython.display as display
import matplotlib.pyplot as plt
import cv2 as cv


def caption_image(image_path):
    # image_path中是绝对路径，将绝对路径处理为仅2层路径
    # D:\...\flower_photos\tulips\5757091018_cdfd79dfa6_m.jpg
    # to
    # tulips\5757091018_cdfd79dfa6_m.jpg
    image_rel = pathlib.Path(image_path).relative_to(data_root)
    print(image_path)
    print(image_rel)
    image_rel = str(image_rel).replace('\\', '/')  # Windows环境路径斜杠转化
    return "Image (CC BY 2.0) " + ' - '.join(attributions[str(image_rel)].split(' - ')[:-1])


for n in range(3):
    image_path = random.choice(all_image_paths)
    img = cv.imread(image_path)
    # plt.subplot(2, 2, n+1)
    plt.xticks([])  # 指定x轴的刻度显示（不显示）
    plt.yticks([])
    plt.grid(False)  # 不绘制网格
    # print(caption_image(image_path))  # 处理图像来源信息
    plt.title(caption_image(image_path), y=-0.1)  # 设置图片下方标题
    plt.imshow(img)
    # plt.show()

# %% 确定图片标签
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
# data_root.glob('*/') 与 data_root.iterdir() 功能相同，取目录下所有文件，ls/dir
# print(list(data_root.glob('*/')))
# print(list(data_root.iterdir()))
label_names

# %% 为每个标签分配索引
# enumerate枚举，为每个值编号，形成元组
# 将元组值放入字典，并将顺序转换为，{标签名：索引}
label_to_index = dict((name, index) for index, name in enumerate(label_names))
label_to_index

# %% 创建列表，包含每个文件的标签索引
# all_image_paths = D:\HomePC\dataset\flower_photos\tulips\5691672942_70a93d70fc.jpg
# path = all_image_paths
# pathlib.Path(path).name = 5691672942_70a93d70fc.jpg
# pathlib.Path(path).parent.name = tulips
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])


# %% 加载和格式化图片
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)  # 读取原始数据
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)  # 将数据解码为张量tensor
    image = tf.image.resize(image, [192, 192])  # 根据模型调整大小
    image /= 255.0  # normalize to [0,1] range

    return image


# %% 测试图片读取
import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]

plt.imshow(load_and_preprocess_image(image_path))
plt.grid(False)
plt.xlabel(caption_image(image_path))
plt.title(label_names[label].title())
plt.show()
print()

# %% 构建一个tf.data.Dataset
# 将字符串数组切片，得到一个字符串数据集
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# shapes维数与type类型，描述了数据集里每个数据项的内容，这里是一组二进制字符串标量
print(path_ds)

# 创建一个新的数据集，在路径数据集上映射preprecess_image来动态加载和格式化图片
# TensorSliceDataset.map(map_func, num_parallel_calls)
# 从切片的字符串数组（路径）中，对对应的图片使用map_func进行处理后映射
# num_parallel_calls表示多线程的线程数
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
for n, image in enumerate(image_ds.take(4)):
    plt.subplot(2, 2, n + 1)
    plt.imshow(image)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(caption_image(all_image_paths[n]))

plt.show()

# %% 使用同样的切片方法得到标签数据集
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(10):
    print(label_names[label.numpy()])

# %% 标签数据集与图片数据集具有相同顺序，使用zip()打包
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print(image_label_ds)

# %% 针对两个有序数组，也可直接打包
ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))


# 元祖元素被分别解压缩，并映射到函数的参数中
def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds

# %% 训练方法
# 打乱训练集、将训练集分割为batch、batch间不重复、batch尽快提供
# 使用 tf.data api 实现上述功能
BATCH_SIZE = 16

# 设置一个和数据集一样大小的 shuffle buffer size （随机缓冲区大小）
# 保证数据被充分打乱

image_count = len(all_image_paths)
ds = image_label_ds.shuffle(buffer_size=image_count)  # 打乱数据
ds = ds.repeat()  # 使数据不断重复
ds = ds.batch(BATCH_SIZE)  # 每次batch取多少数据

ds = ds.prefetch(buffer_size=AUTOTUNE)  # 模型训练时，数据集在后台取得batch
ds

# 打乱时需要注意顺序，repeat之后shuffle，会在epoch之间打乱数据
# 即，会改变每个epoch中的元素，可能导致数据重复出现
# 在batch之后shuffle，会打乱batch的顺序，但不会打乱batch中的数据
# 较大的buffer_size缓冲区提供更好的初始化，但占用更多内存
# 从缓冲区拉去元素前，要先填满缓冲区，缓冲区过大可能引起延迟
# 缓冲区完全为空之前，被打乱的数据集不会反回数据集的结尾
# 故使用repeat重新启动数据集时，缓冲区为空，需要重新填满
# %% 融合shuffle与repeat过程以降低延迟
ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds

# %% 传递数据集至模型
# 使用一个简单的迁移学习
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False

# %% 检查模型需要的输入与输出
import keras_applications

help(keras_applications.mobilenet_v2.preprocess_input)

# %% 将图片映射到对应区间（似乎现在不需要）
# def change_range(image, label):
#     return 2 * image - 1, label
#
#
# keras_ds = ds.map(change_range)

# %% 启动一个batch的数据集
# 数据集可能需要几秒来启动，因为要填满其随机缓冲区。
# image_batch, label_batch = next(iter(keras_ds))
image_batch, label_batch = next(iter(ds))

# %% 传递一个batch的图片给这个模型，查看结果，预期输出(BATCH_SIZE, 6, 6, 1280)
feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

# %% 构建模型，在模型最后添加空间池化与全连接层
model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation='softmax')])

# %% 试验模型
# model.summary()
logit_batch = model(image_batch).numpy()  # 相当于跑了前向传播？

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)
# %% 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

# Dense层中的W与b是可训练的，迁移学习模型中的权重设置为不可训练以加快训练速度
len(model.trainable_variables)
model.summary()

# %% 每个epoch需要进行的迭代次数steps_pre_epoch
# iteration：1次迭代，每次迭代更新1次网络参数（training step）
# batch_size：每次迭代所使用的的样本量
# epoch：1个epoch即遍历1次整个样本

# steps_per_epoch：以现在的batch_size。完成一个epoch需要多少个training step
steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()
steps_per_epoch

# %% 训练模型
model.fit(ds, epochs=1, steps_per_epoch=3)  # 在这里只运行3次迭代，作为演示

# %% 提升性能
# 以上方式使用了简单的pipeline，在每个epoch中单独读取每个文件
# 在进行GPU或分布训练时可能不适用

# 量化性能
import time

default_timeit_steps = 2 * steps_per_epoch + 1


def timeit(ds, steps=default_timeit_steps):
    overall_start = time.time()
    # 开始计时之前
    # 取得单个 batch 填充 pipeline（填充随机缓冲区）
    it = iter(ds.take(steps + 1))  # 创建一个迭代器对象
    next(it)  # 跳过了第一个元素？

    start = time.time()
    for i, (images, labels) in enumerate(it):
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))
    print("Total time: {}s".format(end - overall_start))


# %% 量化性能
ds = image_label_ds.apply(
    tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# %%
timeit(ds)


# %% test
def print_tf(tf_data_dataset):
    i = 0
    for _ in tf_data_dataset:
        if i >= 10:
            break
        tf.print(_)
        i += 1


# steps = 2 * steps_per_epoch + 1
# print(steps_per_epoch)
# print(steps)
# ds.take(1)
# list(ds)
dataset = tf.data.Dataset.range(10)  # 创建一个Dataset
# dataset = dataset.shuffle(buffer_size=10)  # 打乱
dataset = dataset.repeat()  # 重复
# dataset = dataset.apply(  # 将Dataset打乱并不断往复，合并函数，可提升性能
#     tf.data.experimental.shuffle_and_repeat(buffer_size=10))
dataset = dataset.batch(2)  # 每个batch取两个元素，即每次取2个元素
# 从dataset中预先取出元素，在处理当前元素时取出之后的元素
dataset = dataset.prefetch(buffer_size=AUTOTUNE)
# %%
print_tf(next(iter(dataset)))
# it = iter(dataset.take(5))
# print(it)
# next(it)
# print(it)
