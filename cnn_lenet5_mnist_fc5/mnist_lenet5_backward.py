import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#前向传播模块
import mnist_lenet5_forward		
import os
import numpy as np

#一次喂入神经网络多少张图片
BATCH_SIZE = 100
#最开始的学习率
LEARNING_RATE_BASE = 0.005
#学习率衰减率
LEARNING_RATE_DECAY = 0.99
#正则化系数
REGULARIZER = 0.0001
#训练的轮数
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"

#反向传播训练参数：前向传播搭建网络
def backward(mnist):
	# 卷积要求输入四阶张量
	x = tf.placeholder(tf.float32,[
		BATCH_SIZE,		
		mnist_lenet5_forward.IMAGE_SIZE,
		mnist_lenet5_forward.IMAGE_SIZE,
		mnist_lenet5_forward.NUM_CHANNELS])
	y_ = tf.placeholder(tf.float32,[None,mnist_lenet5_forward.OUTPUT_NODE])
	#调用前向传播的程序，输出y
	y = mnist_lenet5_forward.forward(x,True,REGULARIZER)
	#轮数计数器
	global_step = tf.Variable(0,trainable = False)

	#交叉熵：表明两个概率分布之间的距离
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
	cem = tf.reduce_mean(ce)
	#定义损失函数
	loss = cem + tf.add_n(tf.get_collection('losses'))

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY,
		staircase = True)

	#定义训练过程：这个是梯度下降优化程序
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)

	#定义滑动平均
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	#每一次运行这一句，所有待优化参数求滑动平均
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step,ema_op]):
		train_op = tf.no_op(name = 'train')

	saver = tf.train.Saver()

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)

		#加载ckpt模型,实现断点续训	#给所有的w和b赋值保存再ckpt中的值,实现断点续训
		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess,ckpt.model_checkpoint_path)
		#加载完成

		for i in range(STEPS):
			#每次读入BATCH_SIZE组数据
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			# 实现从数据集中的数据进行reshape操作,仅仅调整x
			reshaped_xs = np.reshape(xs,(
				BATCH_SIZE,
				mnist_lenet5_forward.IMAGE_SIZE,
				mnist_lenet5_forward.IMAGE_SIZE,
				mnist_lenet5_forward.NUM_CHANNELS))
			_,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
			if i % 100 == 0:
				print("after %d training steps,loss on training batch is %g" %(step,loss_value))
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
			if i % 10000 == 0:
				#print("查看某个参数的滑动平均值" %(ema.average(参数名)))
				pass
	pass

def main():
	mnist = input_data.read_data_sets('./data/',one_hot = True)
	backward(mnist)

if __name__ == '__main__':
	main()