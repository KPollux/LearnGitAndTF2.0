import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def produce_random_data(u_x, d_x, u_y, d_y, num):
    X1 = np.random.uniform(u_x, d_x, num)
    X2 = np.random.uniform(u_y, d_y, num)
    X = np.vstack((X1, X2))
    return X.transpose()


def collect_boundary_data(model, v_xs):
    # global model
    # X = np.empty([1, 2])
    # X = list()
    # for i in range(len(v_xs)):
    #     x_input = v_xs[i]
    #     x_input.shape = [1, 2]
    #     # y_pre = sess.run(prediction, feed_dict={xs: x_input})
    #     y_pre = model.predict(x_input)
    #     if abs(y_pre - 0) < 0.5:
    #         X.append(v_xs[i])
    y_pre = model.predict(v_xs)
    X = v_xs[np.where(abs(y_pre-0.5) < 0.1)[0]][:]

    return X
    # return 0


def draw_decision_boundary(X_b, data, label, label1, label2):
    # 产生空间随机数据
    # X_NUM = produce_random_data(u_x, d_x, u_y, d_y, num)
    # 边界数据采样
    # X_b = collect_boundary_data(model, X_NUM)

    # 画出数据
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # 设置坐标轴名称
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    plt.scatter(data.loc[label == 1]['Height'],
                data.loc[label == 1]['Weight'], c='blue',
                label=label1)
    plt.scatter(data.loc[label == 0]['Height'],
                data.loc[label == 0]['Weight'], c='red',
                label=label2)
    # 用采样的边界数据拟合边界曲线 7次曲线最佳
    z1 = np.polyfit(X_b[:, 0], X_b[:, 1], 7)
    p1 = np.poly1d(z1)
    x = X_b[:, 0]
    x.sort()
    yvals = p1(x)
    plt.plot(x, yvals, 'r', label='boundray line')
    plt.legend()
    # plt.ion()
    plt.show()


if __name__ == '__main__':
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    dataset_path = "./dataset/student.xls"

    print(dataset_path)
    column_names = ['No', 'Sexual', 'Height', 'Weight']
    raw_dataset = pd.read_excel(dataset_path, names=column_names)  # 重新指定英文列名

    dataset = raw_dataset.copy()  # 复制获取到的数据
    sexual = dataset.pop('Sexual')  # 取出性别列
    dataset['Male'] = (sexual == '男') * 1.0  # 新增男性列，值0表示为女性
    dataset.pop('No')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_stats = train_dataset.describe()  # 包含每列的计数、均值、标准差、最小值、最大值等信息
    train_stats.pop("Male")
    train_stats = train_stats.transpose()
    train_labels = train_dataset.pop('Male')
    test_labels = test_dataset.pop('Male')


    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']


    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    model = tf.keras.models.load_model('./model/sexual.h5')
    X_NUM = produce_random_data(max(normed_train_data['Height']),
                                min(normed_train_data['Height']),
                                max(normed_train_data['Weight']),
                                min(normed_train_data['Weight']),
                                num=5000)

    X_b = collect_boundary_data(model, X_NUM)
    draw_decision_boundary(X_b, normed_train_data, train_labels,
                           label1='male', label2='female')
    # %%
    # arr = X_b[X_b < 0.5]
    # arr.shape
    # np.where(X_b < 1)
    # min(X_b)
    # x = train_stats['mean']
    # x['Height'] = 175
    # x['Weight'] = 65
    # x = np.array([175, 65]).reshape(1, 2)
    # x_dict = {'Height': [175], 'Weight': [65]}
    # x = pd.DataFrame(x_dict)
    # # model.predict(norm(x))
    # # normed_train_data.shape
    # print(x)
    # st = "男" if model.predict(norm(x))[0][0] > 0.5 else "女"
    # print(st)
    #
    # y_pre = model.predict(X_NUM)
    # # %%
    # plt.scatter(X_NUM.transpose()[0], X_NUM.transpose()[1])
    # plt.show()
    # # %%
    # for i, y in enumerate(y_pre):
    #     plot_y = [i, y]
    # plt.scatter(range(len(y_pre)), abs(y_pre-0.5))
    # plt.show()
    #
    # # %%
    # X_NUM[np.where(abs(y_pre-0.5) < 0.1)[0]][:]

