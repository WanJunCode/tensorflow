import tensorflow as tf
import numpy as np

v=tf.Variable(0,dtype=tf.float32,name="v")
ema=tf.train.ExponentialMovingAverage(0.99)

# 读取模型中 v 的影子变量 ， 作为该图中 v 的值
# 使用 ema.variables_to_restore() 可以免去手写 滑动平均模型名称 的步骤
saver=tf.train.Saver(ema.variables_to_restore())
# saver=tf.train.Saver({"v/ExponentialMovingAverage":v})
# saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,"D:/sample/chijiuhua/model_ema.ckpt")
    print(sess.run(v))