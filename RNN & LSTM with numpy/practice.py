import numpy as np
import random
import time
import cllm_utils
import matplotlib.pyplot as plt

from lstm_forward import lstm_forward
from lstm_backward import lstm_backward


def clip(gradients, maxValue):
    """
    梯度修建，避免梯度消失/梯度爆炸

    :param gradients: 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
    :param maxValue: 阈值，把梯度值限制在[-maxValue, maxValue]内
    :return: gradients，修剪后的梯度
    """

    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    return gradients


def sample(parameters, char_to_ix, seed):
    """
    根据RNN输出的概率分布，对字符序列进行采样

    :param parameters: 参数字典，包含Waa, Wax, Wya, by, b
    :param char_to_ix: 字符映射到索引的字典
    :param seed: 随机种子
    :return: indices，包含采样字符索引的长度为n的列表。
    """

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))

    indices = []
    idx = -1        # 用于检测换行符

    counter = 0
    newline_character = char_to_ix["\n"]
    while (idx != newline_character and counter < 50):
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        y = cllm_utils.softmax(np.dot(Wya, a) + by)

        np.random.seed(counter + seed)
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())    # 根据概率分布抽取字符，记录索引
        indices.append(idx)

        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a

        seed += 1
        if counter == 50:
            indices.append(char_to_ix["\n"])

    return indices


def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    执行模型训练的单步优化，SGD

    :param X:   整数列表，其中每个整数映射到词汇表中的字符
    :param Y:   整数列表，与X完全相同，但向左移动了一个索引。
    :param a_prev:  上一个隐藏状态
    :param parameters:  字典，包含各种参数
    :param learning_rate:   学习率
    :return:
                loss -- 损失函数的值（交叉熵损失）
                gradients -- 字典，包含了以下参数：
                        dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a)
                        dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a)
                        db -- 偏置的梯度，维度为(n_a, 1)
                        dby -- 输出偏置向量的梯度，维度为(n_y, 1)
                a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1)
    """

    # 前向传播
    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)

    # 反向传播
    gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)

    # 梯度修剪，[-5 , 5]
    gradients = clip(gradients, 5)

    # 更新参数
    parameters = cllm_utils.update_parameters(parameters, gradients, learning_rate)

    return loss, gradients, a[len(X) - 1]


def model(data, ix_to_char, char_to_ix, num_iterations=3500,
          n_a=50, dino_names=7, vocab_size=27):
    """
    训练模型并生成恐龙名字

    :param data: 语料库
    :param ix_to_char: 索引映射字符字典
    :param char_to_ix: 字符映射索引字典
    :param num_iterations: 迭代次数
    :param n_a: RNN单元数量
    :param dino_names: 每次迭代中采样的数量
    :param vocab_size: 在文本中的唯一字符的数量
    :return: 学习后的参数
    """
    global loss_val

    # 获取维度
    n_x, n_y = vocab_size, vocab_size

    # 初始化参数、损失函数
    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)
    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)

    # 构建名称列表
    with open("./data/dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    np.random.seed(0)
    np.random.shuffle(examples)

    # 初始化隐藏状态
    a_prev = np.zeros((n_a, 1))

    # 循环
    for j in range(num_iterations):
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # 执行单步优化：前向传播 -> 反向传播 -> 梯度修剪 -> 更新参数
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
        # 使用延迟来保持损失平滑. 加速训练。
        loss = cllm_utils.smooth(loss, curr_loss)

        loss_val.append(loss)

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % 2000 == 0:
            print("第" + str(j + 1) + "次迭代，损失值为：" + str(loss))

            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                cllm_utils.print_sample(sampled_indices, ix_to_char)

                # 为了得到相同的效果，随机种子+1
                seed += 1

            print("\n")
    return parameters


if __name__ == '__main__':
    data = open("./data/dinos.txt", "r").read()
    data = data.lower()
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print(chars)
    print("共计有%d个字符，唯一字符有%d个" % (data_size, vocab_size))

    char_to_ix = {ch: i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i: ch for i, ch in enumerate(sorted(chars))}

    loss_val = []

    # 开始时间
    start_time = time.clock()

    # 开始训练
    parameters = model(data, ix_to_char, char_to_ix, num_iterations=10000)

    # 结束时间
    end_time = time.clock()

    # 计算时差
    minium = end_time - start_time

    print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")

    plt.figure()
    plt.plot(loss_val[1000:])
    plt.xlabel('iterations')
    plt.ylabel('loss: cross-entropy')
    plt.show()

    # fig, axes = plt.subplots(2, 1)
    # axes[0].plot(loss_val)
    # axes[1].plot(loss_val[1000:])
    # axes[0].set_xlabel('iterations')
    # axes[0].set_ylabel('loss: cross-entropy')
    # axes[1].set_xlabel('iterations')
    # axes[1].set_ylabel('loss: cross-entropy')
    # fig.show()
