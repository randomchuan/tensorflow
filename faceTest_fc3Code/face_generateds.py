#coding:utf-8
#数据集制作与获取
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

face_train_path = './trainData/labels.txt'
face_test_path = './testData/labels.txt'
tfRecord_train_name = './data/face_train.tfrecords'
tfRecord_test_name = './data/face_test.tfrecords'
data_path = './data'

#****************测试函数***********************
def print_tensor(tensor):
	print("进入了?222")
	sess = tf.Session()
	print(sess.run(tensor))

#****************获取数据集**********************
def get_tfrecord(num = 20,isTrain = True):
	if isTrain:
		tfRecord_path = tfRecord_train_name
	else:
		tfRecord_path = tfRecord_test_name
	feature,label = read_tfRecord(tfRecord_path)
	feature_batch,label_batch = tf.train.shuffle_batch(
		[feature,label],
		batch_size = num,
		num_threads = 2,
		capacity = 40,
		min_after_dequeue = 20)
	return feature_batch,label_batch

def read_tfRecord(tfRecord_path):
	fileName_queue = tf.train.string_input_producer([tfRecord_path])
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(fileName_queue)
	features = tf.parse_single_example(
		serialized_example,
		features = {
		#键名要和训练集的相同，标签里面写入分类个数，这里2分类
		'label':tf.FixedLenFeature([2],tf.int64),
		'features':tf.FixedLenFeature([3],tf.float32)
		})
	# feature = features['features']
	feature = tf.cast(features['features'],tf.float32)
	label = tf.cast(features['label'],tf.int64)
	# feature.reshape([3])
	return feature,label

#****************生成数据集**********************

def write_tfRecord(tfRecord_name,face_path):
	writer = tf.python_io.TFRecordWriter(tfRecord_name)
	num_feature = 0
	f = open(face_path,'r')
	contents = f.readlines()
	f.close()
	for content in contents:
		# 笑脸  特征值1 特征值2 特征值3
		value = content.split()
		labels = [0] * 2
		labels[int(value[0])] = 1
		features = [0]*3
		for i in range(3):
			features[i] = float(value[i+1])
		#封装
		example = tf.train.Example(features = tf.train.Features(feature = {
			'features':tf.train.Feature(float_list = tf.train.FloatList(value = features)),
			'label':tf.train.Feature(int64_list = tf.train.Int64List(value = labels))
			}))
		#序列化
		writer.write(example.SerializeToString())
		num_feature += 1
		if num_feature % 100 == 0:
			print("labels = %s" %(labels))
			print("features = %s" %(features))
	writer.close()
	print("write tfrecord %d successful " % num_feature )

def generate_tfRecord():
	isExists = os.path.exists(data_path)
	if not isExists:
		os.makedirs(data_path)
		print("The directory was created successfully")
	else:
		print("The directory already exists")
	#生成数据集并写入文件
	write_tfRecord(tfRecord_train_name,face_train_path)
	write_tfRecord(tfRecord_test_name,face_test_path)
	pass

def main():
	# generate_tfRecord()
	# features,_ = get_tfrecord(20)	#读取数据出来	
	# print(features)		#有打印
	# print(features.dtype)
	# sess = tf.Session()
	# sess.run(tf.Print(features,[features]))	#没有打印
	# print("执行了？")	#这句话没有执行

	# sess.run(tf.Print(train_logits,[train_logits]))
	# print("111？")
	# print_tensor(features)
	# print_tensor(labels)
	# print("打印了？11111")

	pass

if __name__ == '__main__':
	main()