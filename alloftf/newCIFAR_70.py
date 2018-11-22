import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers

# CIFAR-10 데이터를 다운로드 받기 위한 keras의 helper 함수인 load_data 함수를 임포트합니다.
from tensorflow.keras.datasets.cifar10 import load_data

def next_batch(num, data, labels):
  '''
  `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
  '''
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[ i] for i in idx]
  labels_shuffle = [labels[ i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)
X_img = x
weight_decay = 1e-4

# Convolutional Layer #1
conv1 = tf.layers.conv2d(inputs=X_img, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizers.l2(weight_decay))
# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
dropout1 = tf.layers.dropout(inputs=pool1,rate=keep_prob)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.elu, kernel_regularizer=regularizers.l2(weight_decay))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
dropout2 = tf.layers.dropout(inputs=pool2, rate=keep_prob)

# Convolutional Layer #2 and Pooling Layer #2
conv3 = tf.layers.conv2d(inputs=dropout2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.elu, kernel_regularizer=regularizers.l2(weight_decay))
pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="same", strides=2)
dropout3 = tf.layers.dropout(inputs=pool3, rate=keep_prob)

# Convolutional Layer #4 and Pooling Layer #4
conv4 = tf.layers.conv2d(inputs=dropout3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.elu, kernel_regularizer=regularizers.l2(weight_decay))
pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], padding="same", strides=2)
dropout4 = tf.layers.dropout(inputs=pool4, rate=keep_prob)

# Convolutional Layer #4 and Pooling Layer #4
conv5 = tf.layers.conv2d(inputs=dropout4, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.elu, kernel_regularizer=regularizers.l2(weight_decay))
pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], padding="same", strides=2)
dropout5 = tf.layers.dropout(inputs=pool5, rate=keep_prob)

# Dense Layer with Relu
flat = tf.reshape(dropout5, [-1, 128 * 4])
dense6 = tf.layers.dense(inputs=flat, units=625, activation=tf.nn.relu)
dropout6 = tf.layers.dropout(inputs=dense6, rate=keep_prob)


# Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
logits = tf.layers.dense(inputs=dropout6, units=10)

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

kpp = 0.5

#모델 저장
saver = tf.train.Saver()

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
  # 모든 변수들을 초기화한다. 
  sess.run(tf.global_variables_initializer())
  
  # 10000 Step만큼 최적화를 수행합니다.
  for i in range(2000):
    batch = next_batch(128, x_train, y_train_one_hot.eval())

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
      loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})

      print("반복(Epoch): %d, 트레이닝 데이터 정확도: %f, 손실 함수(loss): %f" % (i, train_accuracy, loss_print))
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: kpp})
  # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.  
  test_accuracy = 0.0
  for i in range(10):
    test_batch = next_batch(500, x_test, y_test_one_hot.eval())
    test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
  test_accuracy = test_accuracy / 10;
  print("테스트 데이터 정확도: %f" % test_accuracy)
  
