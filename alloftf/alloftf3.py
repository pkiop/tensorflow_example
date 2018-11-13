import tensorflow as tf
import matplotlib.pyplot as plt

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

#W_val = []
#cost_val = []
#for i in range(-30, 50):
#        feed_W = i * 0.1
#        curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
#        W_val.append(curr_W)
#        cost_val.append(curr_cost)

#plt.plot(W_val, cost_val)
#plt.show()

# learning_rate = 0.1
# gradient = tf.reduce_mean((W*X - Y) * X)
# descent = W - learning_rate * gradient
# update = W.assign(descent)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

for step in range(21):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)

gvs = optimizer.compute_gradients(cost) # 이 cost 에 맞는 gradient 를 돌려줘
#원하는대로 수정  
apply_gradients = optimizer.apply_gradients(gvs) # 수정해서 적용 가능 

