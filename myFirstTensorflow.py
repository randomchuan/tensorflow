import tensorflow as tf
def test1():
	a = tf.add(3,5)
	sess = tf.Session()
	print(sess.run(a))
	sess.close()
def test2():
	#更高效的写法
	a = tf.add(3,5)
	with tf.Session() as sess:
		print(sess.run(a))
def test3():
	x = 2
	y = 3
	add_op = tf.add(x,y)
	# print(dir(tf))
	mul_op = tf.multiply(x,y)
	useless = tf.multiply(x,add_op)
	pow_op = tf.pow(add_op,mul_op)
	with tf.Session() as sess:
		print(sess.run(useless))



test3()