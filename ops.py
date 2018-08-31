import tensorflow as tf
import numpy as np




def conv_layer(input, size_in, size_out, kernel=[5,5], strides=[2,2], name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([kernel[0], kernel[1], size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out], name="B"))
        conv = tf.nn.conv2d(input, w, strides=[1, strides[0], strides[1], 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act  # TODO -> could potentially return as max_pool but do I want to max_pool the last layer?

def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input, w) + b
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act
