import tensorflow as tf
import tensorflow中文社区.input_data as input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

# 数字越小越接近随机梯度下降，数字越大越接近梯度下降
BATCH_SIZE=100
TRAIN_STEPS=30000

# 学习率
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99

# 正则化 rate
REGULARIZATION_RATE=0.0001

# 滑动平均衰减率
MOVING_AVERAGE_DECAY=0.99

# 给定神经网络的输入和所有参数，计算前向传播结果。
def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    # 当没有使用滑动平均时
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        return tf.matmul(layer1,weight2)+biases2
    else:
        # 滑动平均类使用方法：avg_class.average(X)  X为 w1 b1 , w2 b2
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weight2))+avg_class.average(biases2)

# 训练模型的过程
def train(mnist):
    x_input=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_input=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    weight1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weight2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    global_step=tf.Variable(0,trainable=False)
    # 初始化滑动平均类,指定衰减率,和训练轮数变量
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # 滑动平均类作用在所有 代表神经网络参数的变量上 except(global_step)
    variable_averages_op=variable_averages.apply(tf.trainable_variables())

    # 计算在当前神经网络前向传播的结果，用于计算平滑平均的类为None，即不使用滑动平均
    y_predict=inference(x_input,None,weight1,biases1,weight2,biases2)
    # 计算使用了滑动平均之后的前向传播结果
    average_y_predict=inference(x_input,variable_averages,weight1,biases1,weight2,biases2)

    # 交叉熵：刻画预测值与真实值之间差距的损失函数  使用滑动平均后的结果 average_y_predict
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=average_y_predict,labels=tf.argmax(y_input,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数，正则化系数为0.0001，用于处理过拟合化
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 需要计算正则化损失的参数：weight1 , weight2
    regularization=regularizer(weight1)+regularizer(weight2)
    loss=cross_entropy_mean+regularization

    learning_rate=tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,
                                             # 当前迭代的轮数
                                             global_step=global_step,
                                             # 过完所有的训练数据需要的迭代次数
                                             decay_steps=mnist.train.num_examples/BATCH_SIZE,
                                             decay_rate=LEARNING_RATE_DECAY)

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # 在训练神经网络模型时，每过一遍数据既要 反向传播，又要更新滑动平均值。
    # traon_op=tf.group(train_step,variable_averages_op)   等价
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op=tf.no_op(name='train')

    # 判断两个张量上的没一维是否相等     得到模型在一组数据上的正确率
    correct_prediction=tf.equal(tf.argmax(average_y_predict,1),tf.argmax(y_input,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # 保存模型的地址
    saver = tf.train.Saver()
    model_path = "D:\sample\shouxieNumber\model.ckpt"
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据 validate
        validate_feed={
            x_input:mnist.validation.images,
            y_input:mnist.validation.labels
        }
        # 准备测试数据 test
        test_feed={
            x_input:mnist.test.images,
            y_input:mnist.test.labels
        }

        for i in range(TRAIN_STEPS):
            if i%1000 == 0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average model is "%(i))
                print(validate_acc)
                print(test_acc)
                saver.save(sess,model_path)
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x_input:xs,y_input:ys})

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train(mnist)

if __name__=="__main__":
    tf.app.run()