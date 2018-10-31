from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import tensorflow as tf

print tf.convert_to_tensor(mnist.train.images).get_shape()


