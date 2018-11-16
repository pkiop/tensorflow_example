import tensorflow as tf 

#데이터 가져오기
from tensorflow.examples.tutorials.mnist import input_data

#one_hot true 하면 값 가져올때 알아서 one_hot 처리를 해준다. 
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#한번에 다 올리면 메모리만 많이 차지하고 그럴 필요 없으니까 100개씩 받는다. 
batch_xs, batch_ys = mnist.train.next_batch(100)

nb_classes = 10
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

#softmax => 모델을 0~1사이의 값을 갖는 것으로 변환
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

#test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
#전체 데이터셋을 몇번 학습 시킬까 => epoch 
# epoch 정의 => 전체데이터셋을 한번 학습 시킨 것 
training_epochs = 15
#한번에 몇개를 학습시킬까
batch_size = 100

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
                avg_cost = 0
                # 총 배치를 몇번 돌아야 하는지 계산
                total_batch = int(mnist.train.num_examples / batch_size)

                for i in range(total_batch):
                        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                        c, _ = sess.run([cost, optimizer], feed_dict = {X: batch_xs, Y:batch_ys})
                        avg_cost += c/total_batch
                
                print('Epoch:', '%04d' % (epoch + 1), 'cost = ' , '{:.9f}'.format(avg_cost))

        #accuracy를 sess.run 으로 돌릴 수도 있고 eval 을 이용해서 돌릴 수도 있다. 
        print("Accuracy : ", accuracy.eval(session = sess, feed_dict={X : mnist.test.images, Y: mnist.test.labels}))
