import tensorflow as tf
import read_data


# Parameters.
learning_rate = 0.01
epochs = 10
display_step = 5
batch_size = 100

train_data, train_label, test_data, test_label = read_data.read_data_csv()

# Validation data.
share = 50000
train_x_1 = train_data[:share]
train_y_1 = train_label[:share]
valid_x = train_data[share:]
valid_y = train_label[share:]

train_x = read_data.divide_batches(train_x_1, batch_size)
train_y = read_data.divide_batches(train_y_1, batch_size)

# Input.
x = tf.placeholder(tf.float32, shape=[None, 784], name="Input")

# Output.
y = tf.placeholder(tf.float32, shape=[None, 10], name="Output")

# Weight.
w = tf.Variable(tf.random_normal([784, 10]), name="Weight")

# Bias.
b = tf.Variable(tf.random_normal([10]), name="Bias")

# Model.
with tf.name_scope("Model"):
    model = tf.add(tf.matmul(x, w), b)
    pred = tf.nn.softmax(model)

# Cost.
with tf.name_scope("Cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))

# Optimizer.
with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Accuracy.
with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100

# Create summary for cost and accuracy.
# tf.summary.scalar("Cost", cost)
tf.summary.scalar("Accuracy", accuracy)

# Histogram.
# tf.summary.histogram("weights", w)
# tf.summary.histogram("biases", b)
tf.summary.histogram("accuracy", accuracy)


with tf.name_scope('summaries'):
    mean = tf.reduce_mean(cost)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        # noinspection PyUnresolvedReferences
        stddev = tf.sqrt(tf.reduce_mean(tf.square(cost - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(cost))
    tf.summary.scalar('min', tf.reduce_min(cost))
    tf.summary.histogram('histogram', cost)


# Merge summaries.
merged_summary = tf.summary.merge_all()

# Initialize the variables.
init = tf.global_variables_initializer()

"""
# Save the model.
model_saver = tf.train.Saver()
tf.add_to_collection('x', x)
tf.add_to_collection('y', y)
tf.add_to_collection('accuracy', accuracy)
"""
with tf.Session() as sess:
    sess.run(init)

    # Summary Writer.
    writer = tf.summary.FileWriter("/tmp/mnist_demo/2")
    writer.add_graph(sess.graph)

    # Add run metadata.
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for i in range(1):
        for train, label in zip(train_x, train_y):
            opt = sess.run(optimizer, feed_dict={x: train, y: label})

        s = sess.run(merged_summary, feed_dict={x: valid_x, y: valid_y}, options=run_options, run_metadata=run_metadata)
        writer.add_run_metadata(run_metadata, 'step%d' % i)
        writer.add_summary(s, i)

        print("Epoch - ", i + 1)
        c, acc = sess.run([cost, accuracy], feed_dict={x: valid_x, y: valid_y})
        print("Cost - ", c)
        print("Accuracy - ", acc)
