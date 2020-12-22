import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


if __name__ == '__main__':

    batch_size = 2
    time_step = 4
    n_values = 10
    n_unites = 5
    m = 3

    x_ = np.random.random((batch_size, time_step, n_values))
    # y_ = np.random.random((batch_size, time_step, m))
    y_ = np.array([[[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]],
                   [[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]]])

    # ---- model ----
    inputs = tf.keras.Input(shape=(time_step, n_values))
    h, c, a = layers.LSTM(n_unites, return_sequences=True, return_state=True)(inputs)
    out = layers.TimeDistributed(layers.Dense(m, activation='softmax'), input_shape=(time_step, n_unites))(h)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    opt = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='mean_absolute_error', experimental_run_tf_function=False)
    model.fit(x_, y_, epochs=2)

    pred = model.predict(x_)
    print(pred)

