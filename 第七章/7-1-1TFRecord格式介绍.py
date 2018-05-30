"""
该程序将mnist数据集中所有训练数据全部保存到TFRecord中
"""

import tensorflow as tf
import 手写数字.input_data as input_data
import numpy as np

# 把传入的value转化为整数型的属性，int64_list对应着 tf.train.Example 的定义
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 把传入的value转化为字符串型的属性，bytes_list对应着 tf.train.Example 的定义
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

mnist = input_data.read_data_sets(train_dir = "MNIST_data/",dtype=tf.uint8, one_hot=True)

images=mnist.train.images
labels=mnist.train.labels
pixels=images.shape[1]
num_examples = mnist.train.num_examples

filename_TFRecord="E:/sample/TFRecord/output.tfrecords"
writer=tf.python_io.TFRecordWriter(filename_TFRecord)
for index in range(num_examples):
    # 将图像矩阵转换为一个字符串
    image_raw = images[index].tostring()

    my_dict={'pixels': _int64_feature(pixels),
             'label': _int64_feature(np.argmax(labels[index])),
             'image_raw': _bytes_feature(image_raw)}
    example = tf.train.Example(features=tf.train.Features(feature=my_dict))
    if index==5:
        print(example.SerializeToString())
    writer.write(example.SerializeToString())
    print(index)
writer.close()