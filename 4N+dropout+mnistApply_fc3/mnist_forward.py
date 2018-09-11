import tensorflow as tf

INPUT_NODE = 784	#图片像素是28*28 每个点是0~1的float
OUTPUT_NODE = 10	#十个数，每个数是概率
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

	w3 = get_weight([LAYERL_NODE,OUTPUT_NODE],regularizer)
	b3 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1,w3)+b3
	return y