import numpy as np
import tensorflow as tf
from PIL import Image
import mnist_backward
import mnist_forward

def restore_model(testPicArr):
 	#复现计算图
 	with tf.Graph().as_default() as g:
 		#给x占位
 		x = tf.placeholder(tf.float32,[None,mnist_forward.INPUT_NODE])
 		y = mnist_forward.forward(x,None)
 		preValue = tf.argmax(y,1)	#

 		variable_average = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
 		variable_restore = variable_average.variables_to_restore()
 		saver = tf.train.Saver(variable_restore)

 		with tf.Session() as sess:
 			#加载CKPT
 			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
 			#判断是否有模型
 			if ckpt and ckpt.model_checkpoint_path:
 				#恢复模型到当前会话
 				saver.restore(sess,ckpt.model_checkpoint_path)
 				#执行预测操作
 				preValue = sess.run(preValue,feed_dict = {x:testPicArr})
 				#观察
 				preArr = sess.run(y,feed_dict = {x:testPicArr})
 				preArr = preArr.reshape([1,10])
 				print(preArr)
 				return preValue
 			else:
 				print('No checkpoint file found')
 				return -1

def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28),Image.ANTIALIAS)	#表示用消除锯齿的方法resize
	im_arr = np.array(reIm.convert('L'))		#变成灰度图并转化为矩阵的形式
	threshold = 100
	#因为本来是白底黑字,但是要求是黑底白字,越接近1越白，越接近0越黑
	#二值化处理
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
			if (im_arr[i][j] < threshold):
				im_arr[i][j] = 0
			else:
				im_arr[i][j] = 255
	#拉直
	nm_arr = im_arr.reshape([1,784])
	#转化为浮点数
	nm_arr = nm_arr.astype(np.float32)
	#变成[0,1]之间的浮点数
	img_ready = np.multiply(nm_arr,1.0 / 255.0)
	return img_ready
	pass

def application():
	testNum = input("input the number of test picture : ")	#input是读入数字
	for i in range(int(testNum)):
		testPic = input("the name of test picture : ")	#row_input是实现读入字符串
		#预处理
		testPicArr = pre_pic(testPic)
		preValue = restore_model(testPicArr)
		print("the prediction number is : %d" %preValue)
	pass

def main():
	application()

if __name__ == '__main__':
	main()