#设损失函数loss = (w+1)^2,令w初值是常数10.
#反向传播就是求最优的w，即最小loss对应的w值
#使用指数衰减学习率，在迭代初期得到较高的下降速度
#可以在较小的训练轮数下取得更有效的收敛度
import tensorflow as tf

LEARNING_RATE_BASE = 0.1	#最初的学习率
LEARNING_RATE_DECAY = 0.99	#学习率衰减率
LEARNING_RATE_STEP = 1		#喂入多少轮BATCH_SIZE后，更新一次学习率，一般设置为总样本数/BATCH_ZIZE

#运行几轮BATCHj_ZIZE的计数器，初始为0，设为不被训练
global_step = tf.Variable(0,trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
	global_step,LEARNING_RATE_STEP,LEARNING_RATE_DECAY,staircase = True)
#定义待优化参数w的初始值为5
w = tf.Variable(tf.constant(10,dtype=tf.float32))
#定义损失函数loss
loss = tf.square(w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
#生成会话，训练40轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print("after %s steps:w is %f,loss is %f." %(i,w_val,loss_val))