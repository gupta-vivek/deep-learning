import tensorflow as tf

a = tf.Variable([1, 1])
b = tf.Variable([1, 1])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
    # writer = tf.summary.FileWriter("/tmp/test/3")
    # writer.add_graph(sess.graph)