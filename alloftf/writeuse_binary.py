import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#input placeholders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1,28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = 0.5

W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1), name="w_conv1")
b_conv1 = tf.Variable(tf.constant(0.1,shape=[32]), name="b_conv1")

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1), name="w_conv2")
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]), name="b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1), name="w_fc1")
b_fc1 = tf.Variable(tf.constant(0.1,shape=[1024]), name="b_fc1")

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1), name="w_fc2")
b_fc2 = tf.Variable(tf.constant(0.1,shape=[10]), name="b_fc2")

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

W1_saver = tf.train.Saver({"W_conv1" : W_conv1})
B1_saver = tf.train.Saver({"B_conv1" : b_conv1})

W2_saver = tf.train.Saver({"W_conv2" : W_conv2})
B2_saver = tf.train.Saver({"B_conv2" : b_conv2})

W3_saver = tf.train.Saver({"w_fc1" : w_fc1})
B3_saver = tf.train.Saver({"b_fc1" : b_fc1})

W4_saver = tf.train.Saver({"w_fc2" : w_fc2})
B4_saver = tf.train.Saver({"b_fc2" : b_fc2})

sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

SAVE_DIR = "./model2"

W1_saver.save(sess, SAVE_DIR + '/W1_conv')
B1_saver.save(sess, SAVE_DIR + '/B1_conv')
W2_saver.save(sess, SAVE_DIR + '/W2_conv')
B2_saver.save(sess, SAVE_DIR + '/B2_conv')
W3_saver.save(sess, SAVE_DIR + '/W3_conv')
B3_saver.save(sess, SAVE_DIR + '/B3_conv')
W4_saver.save(sess, SAVE_DIR + '/W4_conv')
B4_saver.save(sess, SAVE_DIR + '/B4_conv')



