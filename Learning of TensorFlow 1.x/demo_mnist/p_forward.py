# -*- coding: utf-8 -*-
import tensorflow as tf


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1 = 500


def get_weights(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer:
        tf.add_to_collection("losses", regularizer(weights))
    return weights


def forward(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weights([INPUT_NODE, LAYER1], regularizer)
        biases = tf.get_variable("biases", [LAYER1], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope("layer2"):
        weights = get_weights([LAYER1, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2

