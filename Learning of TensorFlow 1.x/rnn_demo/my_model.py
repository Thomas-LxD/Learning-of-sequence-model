import tensorflow as tf
import numpy as np
from tensorflow.compat.v1.nn import rnn_cell


class MyModel:

    def __init__(self, batch_size, time_steps, input_size, output_size, hidden_unites, num_layers, is_train=True):

        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_unites = hidden_unites
        self.num_layers = num_layers
        self.is_train = is_train
        self.model_path = './model/model.ckpt'

    def forward(self, input_tensor):
        lstm_cell_list = [rnn_cell.BasicLSTMCell(self.hidden_unites) for _ in range(self.num_layers)]
        cell = rnn_cell.MultiRNNCell(lstm_cell_list)
        h, state = tf.nn.dynamic_rnn(cell, input_tensor, dtype=tf.float32)
        out = h[:, -1, :]
        predict = tf.contrib.layers.fully_connected(out, num_outputs=1, activation_fn=tf.nn.tanh)
        return predict

    def train(self, X, y):

        nums_train = len(X)
        x = np.array(X).swapaxes(1, 2)  # (batch_size, time_step, length)

        inputs = tf.placeholder(tf.float32, shape=(None, self.time_steps, self.input_size), name='x-input')
        outputs = tf.placeholder(tf.float32, shape=(None, self.output_size), name='y-input')

        predict = self.forward(inputs)
        loss = tf.losses.mean_squared_error(labels=outputs, predictions=predict)

        train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(3000):
                start = (i * self.batch_size) % self.batch_size
                end = min(start + self.batch_size, nums_train)
                _, loss_value = sess.run([train_op, loss], feed_dict={inputs: x[start:end], outputs: y[start:end]})
                if i % 500 == 0:
                    saver.save(sess, self.model_path)
                    print('After %d train steps, train loss is %g' % (i, loss_value))

    def predict(self, X, y):
        x = np.array(X).swapaxes(1, 2)
        with tf.Graph().as_default() as g:
            inputs = tf.placeholder(tf.float32, shape=(None, self.time_steps, self.input_size), name='x-input')
            outputs = tf.placeholder(tf.float32, shape=(None, self.output_size), name='y-input')
            predict = self.forward(inputs)
            mse = tf.losses.mean_squared_error(outputs, predict)
            saver = tf.train.Saver()

            with tf.Session() as sess:
                saver.restore(sess, self.model_path)
                mse, res = sess.run([mse, predict], feed_dict={inputs: x, outputs: y})
                print("mse: %g" % mse)
        return res




