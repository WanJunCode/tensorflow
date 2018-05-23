import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

# 第一层卷积层的尺度和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5
# 全连接层的节点个数
FC_SIZE = 512

def inference(input_tensor,train,regularizer):
    with tf.variable_scope('layer1_conv1'):
        conv1_weights=tf.get_variable(name="weight",
                                      shape=[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable(name='bias',
                                     shape=[CONV1_DEEP],
                                     initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input=input_tensor,
                           filter=conv1_weights,
                           strides=[1,1,1,1],
                           padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(value=conv1,
                                        bias=conv1_biases))

    with tf.variable_scope('layer2-pool1'):
        pool1=tf.nn.max_pool(value=relu1,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights=tf.get_variable(name='weight',
                                      shape=[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases=tf.get_variable(name='bias',
                                     shape=[CONV2_DEEP],
                                     initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(input=pool1,
                           filter=conv2_weights,
                           strides=[1,1,1,1],
                           padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(value=conv2,
                                        bias=conv2_biases))

    with tf.variable_scope('layer4-pool2'):
        pool2=tf.nn.max_pool(value=relu2,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME')
        # 获得第二层池化层的shape
        pool_shape=pool2.get_shape().as_list()
        nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]

        reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

    with tf.variable_scope('layer5-fcl'):
        fcl_weights=tf.get_variable(name='weight',
                                    shape=[nodes,FC_SIZE],
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fcl_weights))
        fcl_biases=tf.get_variable(name='bias',
                                   shape=[FC_SIZE],
                                   initializer=tf.constant_initializer(0.1))
        fcl=tf.nn.relu(tf.matmul(reshaped,fcl_weights)+fcl_biases)
        if train:
            fcl=tf.nn.dropout(fcl,0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weight=tf.get_variable(name='weight',
                                   shape=[FC_SIZE,NUM_LABELS],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer !=None:
            tf.add_to_collection('losses',regularizer(fc2_weight))
        fc2_biases=tf.get_variable(name='bias',
                                   shape=[NUM_LABELS],
                                   initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fcl,fc2_weight)+fc2_biases

    return logit