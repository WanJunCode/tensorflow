import tensorflow as tf
import os

tf_record_pattern = os.path.join( 'model/', 'data.tfrecord-*' )
# 查找匹配pattern的文件并以列表的形式返回，filename可以是一个具体的文件名，
# 也可以是包含通配符的正则表达式。
data_files = tf.gfile.Glob( tf_record_pattern )



filename_quene = tf.train.string_input_producer(data_files,shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_quene)

features = tf.parse_single_example(serialized_example,features={
    'i': tf.FixedLenFeature([],tf.int64),
    'j': tf.FixedLenFeature( [], tf.int64),
})


example,lable = features['i'],features['j']
batch_size = 4
capacity = 1000 + 3 * batch_size

example_batch, label_batch = tf.train.batch([example,lable],batch_size = batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    for i in range(15):
        print(sess.run([example_batch,label_batch]))

    print("匹配到的文件名")
    for i in data_files:
        print(i)

    coord.request_stop()
    coord.join(threads)