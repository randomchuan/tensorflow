import tensorflow as tf

INPUT_NODE = 3	#三个特征数据
OUTPUT_NODE = 2	#输出是不是笑脸，1是笑脸，0不是笑脸
LAYERL_NODE = 500	#隐藏层

#权重
def get_weight(shape,regularizer):
	#随机生成w
	w = tf.Variable(tf.truncated_normal(shape,stddev = 0.1))
	#加入正则化
	if regularizer != None:
		tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

#偏置
def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

#前向传播过程
def forward(x,regularizer):
	w1 = get_weight([INPUT_NODE,LAYERL_NODE],regularizer)
	b1 = get_bias([LAYERL_NODE])
	y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

	w2 = get_weight([LAYERL_NODE,LAYERL_NODE],regularizer)
	b2 = get_bias([LAYERL_NODE])
	y2 = tf.nn.relu(tf.matmul(y1,w2)+b2)

	#dropout函数

	w3 = get_weight([LAYERL_NODE,OUTPUT_NODE],regularizer)
	b3 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1,w3)+b3
	return y