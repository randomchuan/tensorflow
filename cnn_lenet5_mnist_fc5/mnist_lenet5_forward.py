import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_NODE = 10

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

# 求卷积
def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding = 'SAME')

# 求最大池化
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

#前向传播过程
def forward(x,train,regularizer):
	conv1_w = get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)
	conv1_b = get_bias([CONV1_KERNEL_NUM])
	conv1 = conv2d(x,conv1_w)
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
	pool1 = max_pool_2x2(relu1)

# 	这一层卷积核的深度是上一层卷积核的个数
	conv2_w = get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
	conv2_b = get_bias([CONV2_KERNEL_NUM])
	conv2 = conv2d(pool1,conv2_w)		#	这一层的输入是上一层的输出
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
	pool2 = max_pool_2x2(relu2)

	pool_shape = pool2.get_shape().as_list()	#输出Pool2的维度，输入到list中
	nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]	#长宽深度
	reshaped = tf.reshape(pool2,[pool_shape[0],nodes])

	fc1_w = get_weight([nodes,FC_SIZE],regularizer)
	fc1_b = get_bias([FC_SIZE])
	fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
	if train:fc1 = tf.nn.dropout(fc1,.5)

	fc2_w = get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
	fc2_b = get_bias([OUTPUT_NODE])
	y = tf.matmul(fc1,fc2_w)+fc2_b
	return y

	# w1 = get_weight([INPUT_NODE,LAYERL_NODE],regularizer)
	# b1 = get_bias([LAYERL_NODE])
	# y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

	# w3 = get_weight([LAYERL_NODE,OUTPUT_NODE],regularizer)
	# b3 = get_bias([OUTPUT_NODE])
	# y = tf.matmul(y1,w3)+b3
	# return y


	# 输出 = tf.nn.dropout(上层输出,暂时舍弃神经元的概率)

	# tf.nn.conv2d(输入描述.eg.[batch,5,5,3]	#batch组数据,每个图片5行5列三通道
	# 	卷积核描述.eg.[3,3,3,16]		#3行3列3通道的卷积核共16个
	# 	核滑动步长.eg.[1,1,1,1]		#1固定,1的行步长,1的列步长,1固定
	# 	padding = 'SAME')		#全零填充	‘VALID'

	# pool = tf.nn.max_pool(输入描述.eg.[batch,28,28,6]	#batch组28*28的图片,6个通道
	# 	池化核描述(仅大小).eg.[1,2,2,1]		#1固定,2行分辨率，2列分辨率，1固定
	# 	池化核滑动步长.eg.[1,2,2,1]		#1固定，2行步长，2列步长，1固定
	# 	padding = 'SAME')