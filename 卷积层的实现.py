import tensorflow as tf
import numpy as np
input_x=tf.placeholder(dtype=np.float32,shape=[28,28,3],name='input_x')

# 前面两个维度代表过滤器的尺寸，第三个维度表示当前层的深度，第四个维度表示过滤器的深度
filter_weight=tf.get_variable(name='weight',
                              shape=[5,5,3,16],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))

biases=tf.get_variable(name='biases',
                       shape=[16],
                       initializer=tf.constant_initializer(0.1))

# 第一个维度表示输入的batch，第二个维
conv=tf.nn.conv2d(input_x,filter_weight,
                  strides=[1,1,1,1],
                  padding='SAME')