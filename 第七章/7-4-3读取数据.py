import tensorflow as tf

# 获取文件列表
files = tf.train.match_filenames_once(pattern="data/data.tfrecords-*")

# 创建输入队列
filename_queue = tf.train.string_input_producer(files,shuffle=False)

# 读取并解析文本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized=serialized_example,
    features={'i': tf.FixedLenFeature([],tf.int64),
              'j': tf.FixedLenFeature([],tf.int64),})

with tf.Session() as sess:
    # 使用 match_filenames_once 需要用local_variables_initializer初始化一些变量
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(8):
        print(sess.run([features['i'],features['j']]))

    # 请求停止
    coord.request_stop()
    # 等待各线程的停止
    coord.join(threads)