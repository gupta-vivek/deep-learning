import tensorflow as tf
import read_data


# Read data.
train_data, train_label, test_data, test_label = read_data.read_data_csv()

with tf.Session() as sess:
    model_saver = tf.train.import_meta_graph('session_c/net3c.sess.meta')
    model_saver.restore(sess, 'session_c/net3c.sess')
    accuracy = tf.get_collection('accuracy')[0]
    x = tf.get_collection('x')[0]
    y = tf.get_collection('y')[0]

    acc = sess.run(accuracy, feed_dict = {x: test_data, y: test_label})

    print("Accuracy - ", acc)
