import numpy as np
import tensorflow as tf
import face_forward
import face_backward

def restore_model(feedData):
 	#复现计算图
 	with tf.Graph().as_default() as g:
 		#给x占位
 		x = tf.placeholder(tf.float32,[None,face_forward.INPUT_NODE])
 		y = face_forward.forward(x,None)
 		preValue = tf.argmax(y,1)	#

 		variable_average = tf.train.ExponentialMovingAverage(face_backward.MOVING_AVERAGE_DECAY)
 		variable_restore = variable_average.variables_to_restore()
 		saver = tf.train.Saver(variable_restore)

 		with tf.Session() as sess:
 			#加载CKPT
 			ckpt = tf.train.get_checkpoint_state(face_backward.MODEL_SAVE_PATH)
 			#判断是否有模型
 			if ckpt and ckpt.model_checkpoint_path:
 				#恢复模型到当前会话
 				saver.restore(sess,ckpt.model_checkpoint_path)
 				#执行预测操作
 				preValue = sess.run(preValue,feed_dict = {x:feedData})
 				#观察
 				preArr = sess.run(y,feed_dict = {x:feedData})
 				# preArr = preArr.reshape([1,10])
 				print(preArr)
 				return preValue
 			else:
 				print('No checkpoint file found')
 				return -1

def application():
	feedData = np.array([[0,0,0]])
	temp = ['first','second','thired']
	for i in range(3):
		feedData[0][i] =(np.float32) (input("input the %s number of feature : " %(temp[i])))	#input是读入数字
	feedData.reshape([1,3])
	preValue = restore_model(feedData)
	print("the prediction number is : %d" %preValue)
	pass

def main():
	while True:
		application()

if __name__ == '__main__':
	main()