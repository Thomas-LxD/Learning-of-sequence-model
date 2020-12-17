import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

data = 'abcdefghijklmnopqrstuvwxyz'
data_set = set(data)
word_2_int = {b: a for a, b in enumerate(data_set)}
int_2_word = {a: b for a, b in enumerate(data_set)}
word_len = len(data_set)
print(word_2_int)
print(int_2_word)


def words_2_ints(words):
    ints = []
    for itmp in words:
        ints.append(word_2_int[itmp])
    return ints


print(words_2_ints('ab'))


def words_2_one_hot(words, num_classes=word_len):
    return keras.utils.to_categorical(words_2_ints(words), num_classes=num_classes)


print(words_2_one_hot('a'))


def get_one_hot_max_idx(one_hot):
    idx_ = 0
    max_ = 0
    for i in range(len(one_hot)):
        if max_ < one_hot[i]:
            max_ = one_hot[i]
            idx_ = i
    return idx_


def one_hot_2_words(one_hot):
    tmp = []
    for itmp in one_hot:
        tmp.append(int_2_word[get_one_hot_max_idx(itmp)])
    return "".join(tmp)


print(one_hot_2_words(words_2_one_hot('adhjlkw')))

time_step = 3  # 一个句子有3个词


def genarate_data(batch_size=5, genarate_num=100):
    # genarate_num = -1 表示一直循环下去,genarate_num=1表示生成一个batch的数据，以此类推
    # 这里，我也不知道数据有多少，就这么循环的生成下去吧。
    # 入参batch_size 控制一个batch 有多少数据，也就是一次要yield进多少个batch_size的数据
    '''
    例如，一个batch有batch_size=5个样本，那么对于这个例子，需要yield进的数据为：
    abc->d
    bcd->e
    cde->f
    def->g
    efg->h
    然后把这些数据都转换成one-hot形式，最终数据，输入x的形式为：

    [第1个batch]
    [第2个batch]
    ...
    [第genarate_num个batch]

    每个batch的形式为：

    [第1句话（如abc）]
    [第2句话（如bcd）]
    ...


    每一句话的形式为：

    [第1个词的one-hot表示]
    [第2个词的one-hot表示]
    ...

    '''
    cnt = 0
    batch_x = []
    batch_y = []
    sample_num = 0
    while (True):
        for i in range(len(data) - time_step):
            batch_x.append(words_2_one_hot(data[i: i + time_step]))
            batch_y.append(words_2_one_hot(data[i + time_step])[0])  # 这里数据加[0]，是为了符合keras的输出数据格式。 因为不加[0]，表示是3维的数据。 你可以自己尝试不加0，看下面的test打印出来是什么
            sample_num += 1
            # print('sample num is :', sample_num)
            if len(batch_x) == batch_size:
                yield (np.array(batch_x), np.array(batch_y))
                batch_x = []
                batch_y = []
                if genarate_num != -1:
                    cnt += 1

                if cnt == genarate_num:
                    return


for test in genarate_data(batch_size=3, genarate_num=1):
    print('--------x:')
    print(test[0])
    print('--------y:')
    print(test[1])

