import tensorflow as tf
from numpy.random import RandomState


def get_data():
    # 生成数据，二分类任务
    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)
    Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
    return X, Y


def get_weight(shape, lamda):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)      # 随机初始化权重
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(lamda)(w))      # 这层参数的正则化项加入计算图集合
    return w


if __name__ == "__main__":

    # 生成数据
    X, Y = get_data()
    print(X)
    print(Y)
    data_size = len(X)

    # 定义batch_size， 越大，梯度方向越准确，训练速度越慢
    batch_size = 16

    # 定义网络结构
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))

    layers_dim = [2, 10, 10, 10, 1]     # 指定每层的节点数
    cur_layer = x       # 用于跟踪当前层
    in_dim = layers_dim[0]
    for i in range(1, len(layers_dim)):          # 循环生成5层深度的NN
        out_dim = layers_dim[i]
        w = get_weight(shape=(in_dim, out_dim), lamda=0.001)
        bias = tf.Variable(tf.random_normal([out_dim]))
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, w) + bias)
        in_dim = layers_dim[i]

    cross_entropy = - tf.reduce_mean(y_ * tf.log(tf.clip_by_value(cur_layer, 1e-10, 1.0)))      # 交叉熵
    tf.add_to_collection("losses", cross_entropy)       # 交叉熵损失 加入集合
    loss = tf.add_n(tf.get_collection("losses"))        # 将集合中的所有损失相加，作为最终损失函数

    # 学习率衰减，学习率过大则不容易收敛，过小则学习速度慢
    learning_rate = tf.train.exponential_decay(0.1, 500, data_size//batch_size, 0.96, False)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        epochs = 500
        for i in range(epochs):
            start = (i * batch_size) % batch_size
            end = min(start + batch_size, data_size)
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

            if i % 100 == 0:        # 每一百次迭代，打印整体损失
                print(sess.run(loss, feed_dict={x: X, y_: Y}))

    



