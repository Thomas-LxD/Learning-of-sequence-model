# -*- coding: utf-8 -*-
import tensorflow as tf
import p_forward
from tensorflow_core.examples.tutorials.mnist import input_data


TRAINING_STEPS = 30000

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
REGULARIZER_RATE = 0.0001

MODEL_PATH = './model/'
MODEL_NAME = 'model.ckpt'


def train(mnist):

    x = tf.placeholder(tf.float32, shape=(None, p_forward.INPUT_NODE), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, p_forward.OUTPUT_NODE), name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_RATE)
    y = p_forward.forward(x, regularizer)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value = sess.run([train_step, loss], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                saver.save(sess, MODEL_PATH+MODEL_NAME)
                valid_loss = sess.run(loss, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                print("After %d steps, train loss is %g, valid loss is %g" % (i, loss_value, valid_loss))


def main(argv=None):
    mnist = input_data.read_data_sets('../mnist_data', one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()


