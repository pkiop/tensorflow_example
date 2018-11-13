import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape = [None, 3])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
        cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X:x_data, Y: y_data})

        if step % 100 == 0:
              print(step, "Cost : ", cost_val, "Prediction:", hy_val)

#학습 시키고 난 후 임의의 사람 점수 확인 

print("Your score will be ", sess.run(hypothesis, feed_dict={X:[[100, 70, 101]]}))

#1번단계
filename_queue = tf.train.string_input_producer(['data-01-test-score.csv'], 'data-02-test-score.csv', ... ], shuffle = False , name = 'filename_queue')
#2번단계 reader 정의 textfile 읽는 기본 
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
#3번단계 어떻게 parsing 할 것인가. record_defaluts => 어떤 데이터 형식인지 제공
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)
#4번단계 batch
train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1]], batch_size = 10)
sess = tf.Session()

#Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord = coord)

for step in range(2001):
        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])

coord.request_stop()
coord.join(threads)

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
example_batch, label_batch = tf.train.suffle_batch([example, label], batch_size= batch_size, capacity=capacity, min_after_dequeue = min_after_dequeue) 
