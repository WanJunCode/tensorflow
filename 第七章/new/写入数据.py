# -*- coding: utf-8 -*-
import tensorflow as tf
import os

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

num_shards = 2
instance_per_shard = 10

for i in range(num_shards):
    filename = 'model/data.tfrecord-%.5d-of%.5d' %(i,num_shards)
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instance_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i':_int64_feature(i),
            'j':_int64_feature(j)
        }))
        writer.write(example.SerializeToString())
    writer.close()


tf_record_pattern = os.path.join( 'model/', 'data.tfrecord-*' )
data_files = tf.gfile.Glob( tf_record_pattern )
filename_quene = tf.train.string_input_producer(data_files,shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_quene)

features = tf.parse_single_example(serialized_example,features={
    'i': tf.FixedLenFeature([],tf.int64),
    'j': tf.FixedLenFeature( [], tf.int64),
})

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)
    for i in range(15):
        print(sess.run([features['i'],features['j']]))

    coord.request_stop()
    coord.join(threads)