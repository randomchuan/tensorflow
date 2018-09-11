#coding:utf-8
#数据集的生成读取
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# image_train_path = './mnist_data_jpg/mnist_train_jpg_60000/'
# label_train_path = './mnist_data_jpg/mnist_train_jpg_60000.txt'
# tfRecord_train = './data/mnist_train.tfrecords'
# image_test_path = './mnist_data_jpg/mnist_test_jpg_10000/'
# label_test_path = './mnist_data_jpg/mnist_test_jpg_10000.txt'
# tfRecord_test = './data/mnist_test.tfrecords'
# data_path = './data'

image_train_path = './trainData/files/'
label_train_path = './trainData/labels.txt'			#这个标签和代码要求的不一样
tfRecord_train = './data/face_train.tfrecords'
image_test_path = './testData/files/'
label_test_path = './testData/labels.txt'		
tfRecord_test = './data/face_test.tfrecords'
data_path = './data'

resize_height = 28
resize_width = 28

#tfRecordName:路劲和文件名	image_path:图像路径	label_path:标签文件
def write_tfRecord(tfRecordName,image_path,label_path):	#tfRecord文件
	writer = tf.python_io.TFRecordWriter(tfRecordName)	#创建一个writer
	num_pic = 0						#计数器
	f = open(label_path,'r')					#读，打开标签文件 label.txt由图片名和标签组成 :	name.jpg 5
	contents = f.readlines()					#读取整个文件的内容
	f.close()							#关闭文件
	for content in contents:					#遍历每一行的内容
		value = content.split()				#分割空格 0是名字,1是标签	返回的是一个列表
		img_path = image_path + value[0]		#得到图片的路径：根目录+图片名
		img = Image.open(img_path)			#打开文件
		img_raw = img.tobytes()			#转换为二进制文件
		labels = [0] * 10					#labels每个元素赋值为0
		labels[int(value[1])] = 1				#正确的那个赋值为1

		#把每张图片和标签封装到example中
		example = tf.train.Example(features = tf.train.Feature(feature={
			'img_raw':tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw])),	#图片
			'label':tf.train.Feature(int64_list = tf.train.Int64List(value=labels))		#他的标签
			}))
		#将example进行序列化
		writer.write(example.SerializeToString())
		num_pic += 1					#保存一个进度加一
		print("the number of priture : " + num_pic)
	writer.close()
	print("write tfrecord successful")

def generate_tfRecord():
	isExists = os.path.exists(data_path)	#判断保存路径
	if not isExists:
		os.makedirs(data_path)
		print("The directory was created successfully")
	else:
		print("directory already exists")
	#把训练集中的图片和标签生成名叫tfRecord_train的文件
	write_tfRecord(tfRecord_train,image_train_path,label_train_path)
	#把测试集中的图片和标签生成名叫tfRecord_test的文件
	write_tfRecord(tfRecord_test,image_test_path,label_test_path)

#读取tfrecord文件
def read_tfRecord(tfRecord_path):		#tfrecord文件
	#新建文件名队列
	filename_queue = tf.train.string_input_producer([tfRecord_path])	#包含哪些文件
	#解序列化
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example,
		features = {
		#键名要和训练集的相同，标签里面写入分类个数，这里是10分类
		'label':tf.FixedLenFeature([10],tf.int64),
		'img_raw':tf.FixedLenFeature([],tf.string)
		})
	#将字符串转换成8位无符号整形
	img = tf.decode_raw(features['img_raw'],tf.uint8)
	img.set_shape([784])					#一行784列
	#变为浮点数
	img = tf.cast(img,tf.float32)*(1./255)			#图片变为浮点数形式
	label = tf.cast(features['label'],tf.float32)		#标签变为浮点数形式
	return img,label

#批获取训练集中的图片和标签，训练集true，测试集false
def get_tfrecord(num,isTrain = True):		#一次多少组
	if isTrain:
		tfRecord_path = tfRecord_train
	else:
		tfRecord_path = tfRecord_test
	img,label = read_tfRecord(tfRecord_path)
	img_batch,label_batch = tf.train.shuffle_batch([img,label],#从总样本中顺序取出capacity组数据，并打乱顺序，每次输出batch_size组
		batch_size = num,
		num_threads = 2,	#两个线程
		capacity = 1000,	#一定会被填满，不够就重复填满
		min_after_dequeue = 700)
	return img_batch,label_batch



def main():
	generate_tfRecord()

if __name__ == '__main__':
	main()

