import numpy as np
import tensorflow as tf
from tensorflow.compat.v1.nn import rnn_cell

from my_model import MyModel


HIDDEN_SIZE = 30        # LSTM隐藏节点个数
NUM_LAYERS = 2          # LSTM层数
TIME_STEPS = 10         # 循环神经网络截断长度
BATCH_SIZE = 32         # batch大小

TRAINING_STEPS = 3000   # 训练轮数
TRAINING_EXAMPLES = 10000   # 训练数据个数
TESTING_EXAMPLES = 1000     # 测试数据个数
SAMPLE_GAP = 0.01           # 采样间隔


def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIME_STEPS - 1):
        X.append([seq[i:i + TIME_STEPS]])
        y.append([seq[i + TIME_STEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":

    # 用sin生成训练和测试数据集
    test_start = TRAINING_EXAMPLES * SAMPLE_GAP
    test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
    train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
    test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

    # lstm_model(train_X, train_y)

    model = MyModel(BATCH_SIZE, TIME_STEPS, 1, 1, HIDDEN_SIZE, NUM_LAYERS)
    model.train(train_X, train_y)
    res = model.predict(test_X, test_y)
    print(test_y[:5])
    print(res[:5])


