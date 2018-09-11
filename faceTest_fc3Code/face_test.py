#为了延迟，导入time模块
import time
import tensorflow as tf
import face_forward
import face_backward
import face_generateds
#程序循环间隔时间
TEST_INTERVAL_SECS = 5
#测试样本数
TEST_NUM = 20

def test(face_data):
 	#复现计算图
 	with tf.Graph().as_default() as g:
 		x = tf.placeholder(tf.float32,[None,face_forward.INPUT_NODE])
 		y_ = tf.placeholder(tf.float32,[None,face_forward.OUTPUT_NODE])
 		y = face_forward.forward(x,None)

 		#恢复过后，数据会得到各自的滑动平均值
 		ema = tf.train.ExponentialMovingAverage(face_backward.MOVING_AVERAGE_DECAY)
 		ema_restore = ema.variables_to_restore()
 		saver = tf.train.Saver(ema_restore)

 		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
 		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

 		#拿数据
 		feature,label = face_generateds.get_tfrecord(TEST_NUM,isTrain = False)
 		
 		while True:
 			with tf.Session() as sess:
 				#滑动平均值赋值给各个参数
 				ckpt = tf.train.get_checkpoint_state(face_backward.MODEL_SAVE_PATH)
 				#判断是否有模型
 				if ckpt and ckpt.model_checkpoint_path:
 					#恢复模型到当前会话
 					saver.restore(sess,ckpt.model_checkpoint_path)
 					#恢复global_step值
 					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
 					#线程调节器
 					coord = tf.train.Coordinator()
 					threads = tf.train.start_queue_runners(sess = sess,coord = coord)

 					xs,ys = sess.run([feature,label])

 					#执行准确率计算
 					accuracy_score = sess.run(accuracy,feed_dict = {x:xs,y_:ys})
 					#打印准确率
 					print("after %s training steps , test accuracy = %g" %(global_step,accuracy_score))
 				else:
 					print('No checkpoint file found')
 					return
 				time.sleep(TEST_INTERVAL_SECS)

def main():
 	#读入数据集
 	#face_data = input_data.read_data_sets('./data/',one_hot = True)
 	test(None)

if __name__ == '__main__':
 	main()