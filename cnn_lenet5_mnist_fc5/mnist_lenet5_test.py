#为了延迟，导入time模块
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np
#程序循环间隔时间
TEST_INTERVAL_SECS = 30

def test(mnist):
 	#复现计算图
 	with tf.Graph().as_default() as g:
 		x = tf.placeholder(tf.float32,[
 			mnist.test.num_examples,
 			mnist_lenet5_forward.IMAGE_SIZE,
 			mnist_lenet5_forward.IMAGE_SIZE,
 			mnist_lenet5_forward.NUM_CHANNELS])
 		y_ = tf.placeholder(tf.float32,[None,mnist_lenet5_forward.OUTPUT_NODE])
 		y = mnist_lenet5_forward.forward(x,False,None)

 		#恢复过后，数据会得到各自的滑动平均值
 		ema = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
 		ema_restore = ema.variables_to_restore()
 		saver = tf.train.Saver(ema_restore)

 		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
 		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

 		while True:
 			with tf.Session() as sess:
 				#滑动平均值赋值给各个参数
 				ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
 				#判断是否有模型
 				if ckpt and ckpt.model_checkpoint_path:
 					#恢复模型到当前会话
 					saver.restore(sess,ckpt.model_checkpoint_path)
 					#恢复global_step值
 					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
 					# reshape_x，这里为什么是mnist.test.num_examples这个参数不是说的全部图片个数吗
 					# 这里mnist.test.images是代表的图片的一个像素，这里是把全部都变成像素点？
 					reshape_x = np.reshape(mnist.test.images,(
 						mnist.test.num_examples,
 						mnist_lenet5_forward.IMAGE_SIZE,
 						mnist_lenet5_forward.IMAGE_SIZE,
 						mnist_lenet5_forward.NUM_CHANNELS))
 					#执行准确率计算
 					accuracy_score = sess.run(accuracy,feed_dict = {x:reshape_x,y_:mnist.test.labels})
 					#打印准确率
 					print("after %s training steps , test accuracy = %g" %(global_step,accuracy_score))
 				else:
 					print('No checkpoint file found')
 					return
 				time.sleep(TEST_INTERVAL_SECS)

def main():
 	#读入数据集
 	mnist = input_data.read_data_sets('./data/',one_hot = True)
 	test(mnist)

if __name__ == '__main__':
 	main()