import tensorflow as tf

#表示定义常数
a = tf.constant([1.0,2.0])
b = tf.constant([3.0,4.0])

result = a+b
print(result)

x = tf.constant([[1,2]])
w = tf.constant([[3],[4]])
#一行一列的张量，得到的不是结果
#矩阵乘法
y = tf.matmul(x,w)
print(y)


#需要得到结果就要用到会话Session()
with tf.Session() as sess:
    #这是要计算结果的运算
    print(sess.run(y))