import numpy as np 
import pprint as pp
import tensorflow as tf 

cost_summ = tf.summary.scalar("cost", cost)

W2 = tf.Variable(tf.random_normal([2,1]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([1]), name = 'bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)


w2_hist = tf.summary.histogram("weights2", W2)
b2_hist = tf.summary.histogran("biases2", b2)
hypothesis_hist = tf.summary.histogram("Hypothesis", hypothesis)

with tf.name_scope("layer1") as scope:
  W1 = tf.Variable(tf.random_normal([2,2]), name = 'weight1')
  b1 = tf.Variable(tf.random_normal([2]), name = 'bias1')
  layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

  w1_hist = tf.summary.histogram("weights1", W1)
  b1_hist = tf.summary.histogram("biases1", b1)
  layer1_hist = tf.summary.histogram("layer1", layer1)

summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph) # Add graph in the tensorboard

s, _ = sess.run([summary, optimizer], feed_dict = feed_dict)
writer.add_summary(s, global_step=global_step)
global_step += 1