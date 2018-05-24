import tensorflow as tf
import numpy as np

# meta 保存了计算图的结构
saver=tf.train.import_meta_graph("D:/sample/chijiuhua/model.ckpt.meta")
with tf.Session() as sess:
    saver.restore(sess,"D:/sample/chijiuhua/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))