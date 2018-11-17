import numpy as np 
import pprint as pp
import tensorflow as tf 

W1 = tf.Variable(tf.random_uniform([2,5], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([5,4], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([4,1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([5]), name = 'Bias1')
b2 = tf.Variable(tf.zeros([4]), name = 'Bias2')
b3 = tf.Variable(tf.zeros([1]), name = 'Bias3')

#Our hypothesis
L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
hypothesis = tf.sigmoid(tf.matmul(L3, W3) + b3)


dropout_rate = tf.placeholder("float")
_L1 = tf.nn.relu(tf.add(tf.matmul(X,W1), B1))
L1 = tf.nn.dropout(_L1, dropout_rate)