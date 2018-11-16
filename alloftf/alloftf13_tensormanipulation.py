import numpy as np 
import pprint as pp
import tensorflow as tf 

t = np.array([[1.,2.,3.], [4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape

sess = tf.Session()
sess.as_default()
t = tf.constant([1,2,3,4]) #1차원 배열이므로 Rank = 1
print(tf.shape(t).eval(session = sess)) #shape 은 배열안의 값이 4개이므로 4

t = tf.constant([[1,2],[3,4]]) #2차원 배열이므로 Rank = 2
print(tf.shape(t).eval(session = sess)) #shape 은 배열안의 값이 2개이므로 2
#rank가 2이니까 output모양은 [2, 2]
t = tf.constant([[[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]]]) #Rank는 2개이므로 2
print(tf.shape(t).eval(session = sess)) #shape 은 배열안의 값이 4개이므로 4
#rank가 4이니까 output모양은 [?, ?, ?, ?]
#제일 안쪽 들어가면 4개
#4개묶음을 하나로 묶으면 또 3개짜리 묶음
#2, 1개 묶음 차례로 볼 수 있다. 

[   
    [  
        [
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20],
            [21,22,23,24]
        ]
    ]
]