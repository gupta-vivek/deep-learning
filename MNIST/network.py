import read_data
import tensorflow as tf


# Parameters.
#learning_rate = 0.01
#epochs = 500
learning_rate = 0.9
epochs = 5
display_step = 5
batch_size = 100

# Read data.
train_data, train_label, test_data, test_label = read_data.read_data_csv()

train_x = read_data.divide_batches(train_data, batch_size)
train_y = read_data.divide_batches(train_label, batch_size)

# Input.
x = tf.placeholder(tf.float32,  [None, 784])
# Output.
y = tf.placeholder(tf.float32, [None, 10])

# Weight.
W = tf.Variable(tf.random_normal([784, 10]))
# Bias.
b = tf.Variable(tf.random_normal([10]))
# Model.
model = tf.add(tf.matmul(x, W), b)

# Prediction.
pred = tf.nn.softmax(model)

# Cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, y))

# Optimizer.
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(0.9).minimize(cost)

# Correct prediction.
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Accuracy.
#accuracy = tf.reduce_mean(correct_pred)*100
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100

init = tf.initialize_all_variables()

print("Testing label")
for i in test_label:
    print("Data - ", i)
    print("Shape - ", i.shape)
    break
print("Length - ", len(test_label))

with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        for train, label in zip(train_x, train_y):
            _  = sess.run(optimizer, feed_dict = {x: train, y: label})

        print("Epoch - ", i + 1)
        acc = sess.run(accuracy, feed_dict = {x: test_data, y: test_label})
        print("Testing Accuracy - ", acc)
        c = sess.run(cost, feed_dict = {x: test_data, y: test_label})
        print("Testing Cost - ", c)
