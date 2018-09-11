#酸奶成本9元，酸奶利润1元
#预测大了损失大，故不要预测大，故生成的模型会少预测一些
import tensorflow as tf
import numpy as np
BATCH_SIZE = 8
SEED = 23455
#成本
COST = 9
#利润
PROFIT = 1

rdm = np.random.RandomState(SEED)
X = rdm.rand(32,2)
Y_ = [[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

#1定义神经网络的输入、参数和输出,定义前向传播过程
x = tf.placeholder(tf.float32,shape=(None,2))
y_ = tf.placeholder(tf.float32,shape=(None,1))
w1 = tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))
y = tf.matmul(x,w1)

#2定义损失函数及反向传播方法
#定义损失函数为MSE,反向传播方法为梯度下降
# loss_mse = tf.reduce_mean(tf.square(y_-y))
#自定义损失函数
loss_mse = tf.reduce_sum(tf.where(tf.greater(y,y_),COST*(y-y_),PROFIT*(y_-y)))
#自定义损失函数二,真的就是我自己写的，不知道对不对#貌似是错的，哈哈哈哈
# ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
cem = tf.reduce_mean(ce)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cem)

#3生成会话,并训练STEPS轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	STEPS = 20000
	for i in range(STEPS):
		start = (i*BATCH_SIZE)%32
		end = start+BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
		if i%500 == 0:
			print("after %d training steps,w1 is :" %(i))
			print(sess.run(w1),"\n")
	print("Final w1 is:\n",sess.run(w1))

