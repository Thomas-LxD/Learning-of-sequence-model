import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


if __name__ == '__main__':
    '''使用功能性API构建神经网络'''
    # -------------------- model ------------------
    num_tags = 12           # Number of unique issue tags
    num_words = 10000       # Size of vocabulary obtained when preprocessing text data
    num_departments = 4     # Number of departments for predictions

    title_input = keras.Input(shape=(None, ), name='title')     # 变长的int序列
    body_input = keras.Input(shape=(None, ), name='body')       # 变长的int序列
    tags_input = keras.Input(shape=(num_tags, ), name='tags')   # 二类标签，size=num_tags

    # embedding - 64 dimensional
    title_features = layers.Embedding(num_words, 64)(title_input)
    body_features = layers.Embedding(num_words, 64)(body_input)

    # LSTM deal with embedding vec
    title_features = layers.LSTM(128, return_sequences=False)(title_features)
    body_features = layers.LSTM(32, return_sequences=False)(body_features)

    # concat three features
    x = layers.concatenate([title_features, body_features, tags_input])

    priority_pred = layers.Dense(1, name="priority")(x)
    department_pred = layers.Dense(num_departments, name="department")(x)

    model = keras.Model(
        inputs=[title_input, body_input, tags_input],
        outputs=[priority_pred, department_pred],
    )
    model.summary()
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={          # 为每个输出分配不同的损耗
            'priority': keras.losses.BinaryCrossentropy(from_logits=True),
            'department': keras.losses.CategoricalCrossentropy(from_logits=True),
        },
        loss_weights=[1.0, 0.2],        # 为每种损失赋予不同权重，调整他们对总loss的贡献
    )

    # ------------------ train -----------------
    # Dummy input data
    title_data = np.random.randint(num_words, size=(1280, 10))
    body_data = np.random.randint(num_words, size=(1280, 100))
    tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

    # Dummy target data
    priority_targets = np.random.random(size=(1280, 1))
    dept_targets = np.random.randint(2, size=(1280, num_departments))

    model.fit(
        {'title': title_data, 'body': body_data, 'tags': tags_data},
        {'priority': priority_targets, 'department': dept_targets},
        epochs=2,
        batch_size=32,
    )



