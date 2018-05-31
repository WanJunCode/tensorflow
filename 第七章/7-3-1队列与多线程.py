import tensorflow as tf

# 创建一个先进先出队列
q = tf.FIFOQueue(capacity=3,dtypes="int32")

# 初始化队列中的元素
init = q.enqueue_many(vals=([0,10,5],))

# 出列
x = q.dequeue()

y = x+1

# 将加1后的值重新加入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(10):
        v=sess.run([x, q_inc])
        print(v)