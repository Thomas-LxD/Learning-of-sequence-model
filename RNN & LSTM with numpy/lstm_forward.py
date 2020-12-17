import numpy as np
import rnn_utils


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    实现一个LSTM单元的前向传播。

    参数：
        xt -- 在时间步“t”输入的数据，维度为(n_x, m)
        a_prev -- 上一个时间步“t-1”的隐藏状态，维度为(n_a, m)
        c_prev -- 上一个时间步“t-1”的记忆状态，维度为(n_a, m)
        parameters -- 字典类型的变量，包含了：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)
    返回：
        a_next -- 下一个隐藏状态，维度为(n_a, m)
        c_next -- 下一个记忆状态，维度为(n_a, m)
        yt_pred -- 在时间步“t”的预测，维度为(n_y, m)
        cache -- 包含了反向传播所需要的参数，包含了(a_next, c_next, a_prev, c_prev, xt, parameters)

    注意：
        ft/it/ot表示遗忘/更新/输出门，cct表示候选值(c tilda)，c表示记忆值。
    """

    wf = parameters['Wf']
    bf = parameters['bf']
    wi = parameters['Wi']
    bi = parameters['bi']
    wc = parameters['Wc']
    bc = parameters['bc']
    wo = parameters['Wo']
    bo = parameters['bo']
    wy = parameters['Wy']
    by = parameters['by']

    n_x, m = xt.shape
    n_y, n_a = wy.shape

    contact = np.zeros([n_a + n_x, m])
    contact[:n_a, :] = a_prev
    contact[n_a:, :] = xt

    ft = rnn_utils.sigmoid(np.dot(wf, contact) + bf)
    it = rnn_utils.sigmoid(np.dot(wi, contact) + bi)
    cct = np.tanh(np.dot(wc, contact) + bc)
    c_next = ft * c_prev + it * cct
    ot = rnn_utils.sigmoid(np.dot(wo, contact) + bo)
    a_next = ot * np.tanh(c_next)
    yt_pred = rnn_utils.softmax(np.dot(wy, a_next) + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    实现LSTM单元组成的的循环神经网络

    参数：
        x -- 所有时间步的输入数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为(n_a, m)
        parameters -- python字典，包含了以下参数：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y -- 所有时间步的预测值，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """

    caches = []

    n_x, m, t_x = x.shape
    n_y, n_a = parameters['Wy'].shape

    a = np.zeros([n_a, m, t_x])
    c = np.zeros([n_a, m, t_x])
    y = np.zeros([n_y, m, t_x])

    a_prev = a0
    c_prev = np.zeros([n_a, m])

    for t in range(t_x):
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_prev, c_prev, parameters)
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt_pred
        caches.append(cache)
    caches = (caches, x)

    return a, y, c, caches


if __name__ == '__main__':
    np.random.seed(1)
    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    parameters = {
        'Wf': np.random.randn(5, 5+3),
        'bf': np.random.randn(5, 1),
        'Wi': np.random.randn(5, 5+3),
        'bi': np.random.randn(5, 1),
        'Wc': np.random.randn(5, 5+3),
        'bc': np.random.randn(5, 1),
        'Wo': np.random.randn(5, 5+3),
        'bo': np.random.randn(5, 1),
        'Wy': np.random.randn(2, 5),
        'by': np.random.randn(2, 1),
    }
    a, y, c, caches = lstm_forward(x, a0, parameters)
    print("a[4][3][6] = ", a[4][3][6])
    print("a.shape = ", a.shape)
    print("y[1][4][3] =", y[1][4][3])
    print("y.shape = ", y.shape)
    print("caches[1][1[1]] =", caches[1][1][1])
    print("c[1][2][1]", c[1][2][1])
    print("len(caches) = ", len(caches))
