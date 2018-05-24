import tensorflow as tf
from tensorflow.python.framework import graph_util

v1=tf.Variable(tf.constant(1.0,shape=[1]),name="v1")
v2=tf.Variable(tf.constant(2.0,shape=[1]),name="v2")
result=v1+v2

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # 导出当前图的  GraphDef  部分
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中的变量及其取值转化为常量，add 给出了需要保存的 节点名称
    output_graph_def = graph_util.convert_variables_to_constants(sess=sess,
                                                                 input_graph_def=graph_def,
                                                                 output_node_names=['add'])

    # 将导出的模型存入文件
    with tf.gfile.GFile("D:/sample/chijiuhua/model_constants.pb","wb") as f:
        f.write(output_graph_def.SerializeToString())