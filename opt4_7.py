# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:00:54 2018

@author: Random
"""

#可视化坐标点
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

BATCH_SIZE = 30
seed = 2
#基于seed产生随机数
rdm = np.random.RandomState(seed)
X = rdm.randn(300,2)
#平方和小于2赋值为1，否则赋值为0
Y_ = [int(x0*x0+x1*x1 < 2) for (x0,x1) in X]
#进行可视化赋值，1为red，0为blue
Y_c = [['red' if x0 ==1 else 'blue'] for x0 in Y_]
#对数据集X和标签Y进行shape整理，第一个元素-1表示，随第二个参数计算得到，第二个元素表示多少列，把X整理为n行2列，把Y整理成n行1列
X = np.vstack(X).reshape(-1,2)
Y_ = np.vstack(Y_).reshape(-1,1)
#print(X,Y_,Y_c)

#用plt.scatter画出数据集X各行中第0列元素和第一列元素的点，用对应的Y_c表示颜色
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))#X[:,0]表示取第一列元素
plt.show()

#定义神经网络的输入输出和前向传播过程
def get_weight(shape,regularizer):  #正则化权重
    w = tf.Variable(tf.random_normal(shape),dtype = tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01,shape=shape))
    return b

x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))

w1 = get_weight([2,11],0.01)
b1 = get_bias([11])
y1 = tf.nn.relu(tf.matmul(x,w1)+b1) #relu是非正则化函数

w2 = get_weight([11,1],0.01)
b2 = get_bias([1])
y = tf.matmul(y1,w2)+b2    #输出层不过激活

#定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection('losses'))

#定义反向传播方法：不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%300
        end = start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%2000 == 0:
            loss_mse_v = sess.run(loss_mse,feed_dict={x:X,y_:Y_})
            print("after %d train steps,loss is:%f" %(i,loss_mse_v))
    #xx在-3到3之间以步长为0.01，yy在-3到3之间以补偿0.01，生成二位网格坐标点
    xx,yy = np.mgrid[-3:3:0.01,-3:3:.01]
    #将xx,yy拉直为坐标集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    #将网络坐标点喂入神经网络，probs为输出
    probs = sess.run(y,feed_dict={x:grid})
    #probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:\n",sess.run(w1))
    print("b1:\n",sess.run(b1))
    print("w2:\n",sess.run(w2))
    print("b2:\n",sess.run(b2))
    
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])
plt.show()
    
#定义反向传播方法：包含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%300
        end = start+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%2000 == 0:
            loss_v = sess.run(loss_total,feed_dict={x:X,y_:Y_})
            print("after %d train steps,loss is:%f" %(i,loss_v))
    #xx在-3到3之间以步长为0.01，yy在-3到3之间以补偿0.01，生成二位网格坐标点
    xx,yy = np.mgrid[-3:3:0.01,-3:3:.01]
    #将xx,yy拉直为坐标集合
    grid = np.c_[xx.ravel(),yy.ravel()]
    #将网络坐标点喂入神经网络，probs为输出
    probs = sess.run(y,feed_dict={x:grid})
    #probs的shape调整成xx的样子
    probs = probs.reshape(xx.shape)
    print("w1:\n",sess.run(w1))
    print("b1:\n",sess.run(b1))
    print("w2:\n",sess.run(w2))
    print("b2:\n",sess.run(b2))
    
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.contour(xx,yy,probs,levels=[.5])    #ptobs为0.5的上色
plt.show()















