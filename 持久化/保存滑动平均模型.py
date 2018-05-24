import tensorflow as tf

v=tf.Variable(0,dtype=tf.float32,name='v')
for variables in tf.global_variables():
    print(variables.name)
ema=tf.train.ExponentialMovingAverage(0.99)
maintain_average_op=ema.apply(tf.global_variables())

print()
for variables in tf.global_variables():
    print(variables.name)

saver=tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    sess.run(tf.assign(v,10))
    # 运行滑动平均
    sess.run(maintain_average_op)
    # 保存时会将 v:0 和 v/ExponentialMovingAverage:0 两个变量都保存下来
    saver.save(sess,"D:/sample/chijiuhua/model_ema.ckpt")
    print(sess.run([v,ema.average(v)]))