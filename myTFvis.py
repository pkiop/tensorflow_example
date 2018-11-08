from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

import tensorflow as tf

print(tf.convert_to_tensor(mnist.train.images).get_shape())

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='y-input')

W = tf.Variable(tf.zeros([784,10]), name = 'Weight')
b = tf.Variable(tf.zeros([10]), name = 'bias')

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train' ,sess.graph)
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
tf.global_variables_initializer().run()

for i in range(1000):
    batch = mnist.train.next_batch(50)
    feed_dict = {x:batch[0], y_ : batch[1]}
    train_step.run(feed_dict=feed_dict)

train_writer.close()
test_writer.close()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))






