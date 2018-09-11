#生成tfrecords文件
writer = tf.python_io.TFTecordWriter(tfRecordName)  #新建一个writer
for 循环遍历每张图和标签:
    #把每张图片和标签封装到example中
    example = tf.train.Example(features = tf.train.Features(feature = { #特征以字典的形式给出
            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_row])), #放入二进制的图片文件
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=labels))}))    #放入图片文件的标签
    writer.write(example.SerializeToString())   #把example进行序列化
    writer.close()
    #到此为止就生成了tfrecords文件
    
#解析tfrecords文件
filename_queue = tf.train.string_input_producer([tfTrcord_path])    #文件队列名
reader = tf.TFRecordReader()    #新建一个reader
_,serialize_example = reader.read(filename_queue)
#下面进行解序列化
feature = tf.parse_single_example(serialized_example,features={
        'img_row':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([10],tf.int64)})
img = tf.decode_row(feature['img_row'],tf.uint8)
img.set_shape([784])    #变成一行784列
img.tf.cast(img,tf.float32)*(1./255)    #把每个元素变为浮点数,且位于[0,1]
label = tf.cast(features['label'],tf.float32)   


