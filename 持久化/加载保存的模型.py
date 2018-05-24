import tensorflow as tf
import numpy as np

v1=tf.Variable(tf.constant(1.0,shape=[1]),name="other-v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="other-v2")

result=v1+v2

# 制定了 模型中的 v1 和 本程序中定义的 v1 联系起来
saver=tf.train.Saver({"v1":v1,"v2":v2})

with tf.Session() as sess:
    # 加载模型 ， 将 model.ckpt -> sess  保存了每个变量的取值
    saver.restore(sess,"D:/sample/chijiuhua/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))