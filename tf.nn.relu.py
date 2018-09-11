import tensorflow as tf

a = tf.constant([[-2,-4],[4,-2]])
with tf.Session() as sess:
 	print(sess.run(tf.nn.relu(a)))