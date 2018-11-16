import numpy as np 
import pprint as pp
import tensorflow as tf 

sess = tf.Session()
matrix1 = tf.constant([[1.,2.],[3.,4.]])
matrix2 = tf.constant([[1.],[2.]])

print("Metrix 1 shape", matrix1.shape)
print("Metrix 2 shape", matrix2.shape)

print(tf.matmul(matrix1, matrix2).eval(session = sess))
print((matrix1 * matrix2).eval(session = sess))

print(tf.reduce_mean([1,2], axis = 0).eval(session = sess))

x = [[1., 2.],[ 3., 4.]]

print(tf.reduce_mean(x).eval(session = sess))
print(tf.reduce_mean(x, axis = 0).eval(session = sess)) # [1,3의평균, 2,4의 평균]
print(tf.reduce_mean(x, axis = 1).eval(session = sess)) # [1,2의평균, 3,4의 평균]
print(tf.reduce_mean(x, axis = -1).eval(session = sess))# 가장 안쪽의 평균 가장 많이 사

t = np.array([[[0,1,2], [3,4,5]],[[6,7,8],[9,10,11]]])
print(t.shape)
#(2, 2, 3)

print(tf.reshape(t, shape=[-1,3]).eval(session = sess))
"""
[[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]
"""
print(tf.reshape(t, shape=[-1,1,3]).eval(session = sess))
"""
[[[ 0  1  2]]

 [[ 3  4  5]]

 [[ 6  7  8]]

 [[ 9 10 11]]]
"""

print(tf.squeeze([[0],[1],[2]]).eval(session = sess))
"""
[0 1 2]
"""
print(tf.expand_dims([0,1,2], 1).eval(session = sess))
"""
[[0]
 [1]
 [2]]
"""

print(tf.one_hot([[0], [1], [2], [0]], depth = 3).eval(session = sess))
"""
  [[[1. 0. 0.]]

 [[0. 1. 0.]]

 [[0. 0. 1.]]

 [[1. 0. 0.]]]
"""
t = tf.one_hot([[0], [1], [2], [0]], depth = 3)
print(tf.reshape(t, shape = [-1, 3]).eval(session = sess))
"""
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [1. 0. 0.]]
"""

print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval(session = sess))
# [1 2 3 4]

print(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval(session = sess))
# [1 0 1 0]

print("stack")
x = [1,4]
y = [2,5]
z = [3,6]

print(tf.stack([x,y,z]).eval(session = sess))
"""
[[1 4]
 [2 5]
 [3 6]]
"""
print(tf.stack([x,y,z], axis = 1).eval(session = sess))
"""
[[1 2 3]
 [4 5 6]]
"""