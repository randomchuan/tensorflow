#coding: utf-8
"""
Created on Wed May 16 17:05:25 2018

@author: Random
"""
#两层简单神经网络(全连接)
import tensorflow as tf

#定义输入和参数
x = tf.constant([[0.7,0.5]])
#正态分布，产生2*3的矩阵，标准差为1，种子为1
w1 = tf.Variable(tf.random_normal([2,3],stddev = 1,seed = 1))
w2 = tf.Variable(tf.random_normal([3,1],stddev = 1,seed = 1))

#定义前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#用会话计算结果
with tf.Session() as sess:
    #初始化节点
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in this is :\n",sess.run(y))


