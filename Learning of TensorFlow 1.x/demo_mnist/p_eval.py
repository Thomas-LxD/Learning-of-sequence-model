import time
import tensorflow as tf
import p_forward
import p_train
from tensorflow_core.examples.tutorials.mnist import input_data

EVAL_INTERVAL_SECS = 5


def evaluate(mnist):

    with tf.Graph().as_default() as g:

        x = tf.placeholder(tf.float32, shape=(None, p_forward.INPUT_NODE), name='x-input')
        y_ = tf.placeholder(tf.float32, shape=(None, p_forward.OUTPUT_NODE), name='y-input')

        regularizer = tf.contrib.layers.l2_regularizer(p_train.REGULARIZER_RATE)
        y = p_forward.forward(x, regularizer)

        correct_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

        saver = tf.train.Saver()
        valid_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        while True:
            with tf.Session() as sess:
                saver.restore(sess, p_train.MODEL_PATH + p_train.MODEL_NAME)
                acc = sess.run(accuracy, feed_dict=valid_feed)
                print("Acc : %g" % acc)

            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('../mnist_data', one_hot=True)
    evaluate(mnist)


if __name__ == "__main__":
    tf.app.run()




