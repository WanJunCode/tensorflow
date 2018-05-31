import tensorflow as tf

queue = tf.FIFOQueue(100,"float")

enqueue_op = queue.enqueue(vals=[tf.random_normal(shape=[1])])

# queuerunner
# 使用多个线程完成 队列 queue 的入队操作
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)

# 将 queuerunner 加入 默认的集合中
tf.train.add_queue_runner(qr=qr,
                          collection=tf.GraphKeys.QUEUE_RUNNERS)

# 定义出队操作
out_tensor = queue.dequeue()

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    # 启动所有线程 使用coord 协同线程
    threads = tf.train.start_queue_runners(sess=sess,
                                           coord=coord,
                                           collection=tf.GraphKeys.QUEUE_RUNNERS)
    for _ in range(5):
        print(sess.run(out_tensor)[0])
    
    coord.request_stop()
    coord.join(threads)