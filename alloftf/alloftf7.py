import tensorflow as tf 

#hypothesis = tf.nn.softmax(tf.matmul(X,W) + b)

#합을 하고 평균을 낸다. 
#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

x_data = [[1,2,1,1], [2,1,3,2], [3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
#2, 2, 2, 1, 1, 1, 0, 0, 0 의미한다. 
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

#넣는 것의 y 숫자 => lable의 개수 
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4,nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]),name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(2001):
                sess.run(optimizer, feed_dict = {X:x_data, Y : y_data})
                if step % 200 == 0:
                        print(step, sess.run(cost, feed_dict = {X:x_data, Y:y_data}))

        #Testing & One-hot encoding
        print("test")
        a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
        #argmax
        #[2.5533700e-01 7.4465555e-01 7.4568266e-06] 이런 list에서 몇번째에 있는게 가장 큰지 궁금할때 
        print(a, sess.run(tf.argmax(a,1)))
        all = sess.run(hypothesis, feed_dict={X:[[1,11,7,9], [1,3,4,3], [1,1,0,1]]})
        print(all, sess.run(tf.argmax(all,1)))