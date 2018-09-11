#为了延迟，导入time模块
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward
import mnist_generateds
#程序循环间隔时间
TEST_INTERVAL_SECS = 5

TEST_NUM = 10000	#测试总样本数#mnist.test.num_examples

def test(mnist):
 	#复现计算图
 	with tf.Graph().as_default() as g:
 		x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
 		y_ = tf.placeholder(tf.float32,[None,mnist_forward.OUTPUT_NODE])
 		y = mnist_forward.forward(x,None)

 		#恢复过后，数据会得到各自的滑动平均值
 		ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
 		ema_restore = ema.variables_to_restore()
 		saver = tf.train.Saver(ema_restore)

 		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
 		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

 		#拿到数据，因为是测试集中取出数据，所以istrain为False
 		img_batch,label_batch = mnist_generateds.get_tfrecord(TEST_NUM,isTrain = False)

 		while True:
 			with tf.Session() as sess:
 				#滑动平均值赋值给各个参数
 				ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
 				#判断是否有模型
 				if ckpt and ckpt.model_checkpoint_path:
 					#恢复模型到当前会话
 					saver.restore(sess,ckpt.model_checkpoint_path)
 					#恢复global_step值
 					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
 					#县城协调器
 					coord = tf.train.Coordinator()
 					threads = tf.train.start_queue_runners(sess = sess,coord = coord)

 					xs,ys = sess.run([img_batch,label_batch])

 					#执行准确率计算
 					accuracy_score = sess.run(accuracy,feed_dict = {x:xs,y_:ys})
 					#打印准确率
 					print("after %s training steps , test accuracy = %g" %(global_step,accuracy_score))

 					coord.request_stop()
 					coord.join(threads)
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