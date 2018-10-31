from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

import tensorflow as tf

# y = W*x + b


#matrix W
#to make simple, init with random noise
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#bias B
#to make simple, init with small positive number (0,1)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#for CNN
#stride => sliding window's one move size 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#for pooling()
#size 2x2 pooling 
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding='SAME')
    

#dtype = float, 3x3 0x784 matrix
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

x_image = tf.reshape(x,[-1,28,28,1])

#define bias for 32 Weight maxrix
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])



