import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def model(Tx, n_a, n):
    """
    Define a model

    :param Tx: time step
    :param n_a: dim of hidden state
    :param n: nums of feature / length of each input vector
    :return: model which is before compile
    """
    # define input shape
    X = tf.keras.Input((Tx, n))

    # Initialize state
    a0 = tf.keras.Input(shape=(n_a,), name="a0")
    c0 = tf.keras.Input(shape=(n_a,), name="c0")
    a = a0
    c = c0

    # save all outputs
    outputs = []

    # Loop
    for t in range(Tx):
        # select 't' time step input data
        x = layers.Lambda(lambda x: X[:, t, :])(X)

        # reshape x
        x = layers.Reshape((1, n))(x)

        # ont step
        a, _, c = layers.LSTM(n_a, return_state=True)(x, initial_state=[a, c])

        # for binary
        out = layers.Dense(1, activation='sigmoid')(a)

        # save one step output
        outputs.append(out)

    model = tf.keras.Model(inputs=[X, a0, c0], outputs=outputs)

    return model


if __name__ == '__main__':

    n = 5
    m = 1
    Tx = 4
    batch_size = 1
    n_a = 4   # n dim of hidden state

    x_ = np.random.random((batch_size, Tx, n))
    y_ = np.array([[[1]], [[0]], [[1]], [[0]]])
    # print(x_)

    # acquire model
    model = model(Tx=Tx, n_a=n_a, n=n)

    opt = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'], experimental_run_tf_function=False)

    # Initialize state in 0
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))

    model.fit([x_, a0, c0], list(y_), epochs=2)

    # predict test
    temp = model.predict([x_, a0, c0])


