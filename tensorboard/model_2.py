import tensorflow as tf

a = tf.constant(2.0, name='a')
b = tf.constant(3.0, name='b')
c = tf.multiply(a, b, name='c')

tf.summary.scalar('c',c)
merged = tf.summary.merge_all()

with tf.Session() as sess:

    writer = tf.summary.FileWriter("/tmp/test/2")
    writer.add_graph(sess.graph)

    summary_tf = sess.run(merged)

    # writer.close()
    writer.add_summary(summary_tf)