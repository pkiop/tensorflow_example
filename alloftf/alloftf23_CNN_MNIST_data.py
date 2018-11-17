import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray')

sess = tf.InteractiveSession()


img = img.reshape(-1,28,28,1) #28x28의 한 색깔의 이미지라는 뜻 첫번째 -1이라는 건 list 에서 -1이 전부를 뜻하는 것 처럼 어떤숫자든 와도 된다는 뜻 
W1 = tf.Variable(tf.random_normal([3,3,1,5], stddev = 0.01))
conv2d = tf.nn.conv2d(img, W1, strides = [1,2,2,1], padding = 'SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval(session = sess)

conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap = 'gray')

pool = tf.nn.max_pool(conv2d, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
print(pool)

sess.run(tf.global_variables_initializer())
pool_img = pool.eval(session = sess)
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')

plt.show()
