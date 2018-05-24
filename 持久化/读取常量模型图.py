import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "D:/sample/chijiuhua/model_constants.pb"
    with gfile.FastGFile(model_filename,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将保存的图加载到当前的图中，得到了返回的张量的名称
    result=tf.import_graph_def(graph_def=graph_def,return_elements=["add:0"])
    print(sess.run(result))