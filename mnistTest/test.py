

def test(mnist):
	with tf.Graph().as_default() as g:
		#定义x,y_,y
		#实例化可还原滑动平均值的saver
		#计算正确率
		while True:
			with tf.Session() as sess:
				#加载ckpt模型 
				ckpt = tf.train.get_checkpoint_state(存储路径)
				#如果已有ckpt模型则恢复 
				if ckpt and ckpt.model_checkpoint_path:
				#恢复会话 
				saver.restore(sess,ckpt.model_checkpoint_path)
				#恢复轮数 
				global_step = ckpt.model)checkpoint_path.split('/')[-1].split('-')[-1]
				#计算准确率 
				accuracy_score = sess.run(accuracy,feed_dict = 
						{x:mnist.test.images,y_:mnist.test.labels})
				#打印提示 
				print("after %s training steps , test accurary = %g" %(global_step,accuracy_score))
				#如果没有模型 
			else:
				#给出提示 
				print("No checkpoint file found")
				return
def main():
	mnist = input_data.read_data_sets("./data/",one_hot = True)
	test(mnist)
if __name__ == '__main__':
	main()
