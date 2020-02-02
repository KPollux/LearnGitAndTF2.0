"""
1 使用tf.data加载图片
"""
# %% 导包并下载所需数据
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE  # 可以让程序自动的选择最优的线程并行个数

# %%
import pathlib

data_root_orig = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                         fname='flower_photos', untar=True,
                                         cache_subdir="D:\\HomePC\\dataset")
data_root = pathlib.Path(data_root_orig)
print(data_root)

# %% 检查下载目录是否正常解压
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

# %% 检查图片
import os


