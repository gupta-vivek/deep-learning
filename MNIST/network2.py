import read_data
import tensorflow as tf


# Parameters.
learning_rate = 0.5
epochs = 10
display_step = 5
batch_size = 100

# Network.
input_layer = 784
hidden_layer = 256
output_layer = 10

# Read data.
train_data, train_label, test_data, test_label = read_data.read_data_csv()

train_x = read_data.divide_batches(train_data, batch_size)
train_y = read_data.divide_batches(train_label, batch_size)

# Input.
x = tf.placeholder(tf.float32, [None, 784])

# Output.
y = tf.placeholder(tf.float32, [None, 10])

# Model.
def model(x, weights, biases):

    layer1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    #layer1 = tf.sigmoid(layer1)
    #layer1 = tf.nn.relu(layer1)
    layer1 = tf.tanh(layer1)
    output = tf.add(tf.matmul(layer1, weights['out']), biases['out'])

    return output

# Weights.
weights = {

    'w1': tf.Variable(tf.random_normal([input_layer, hidden_layer])),
    'out': tf.Variable(tf.random_normal([hidden_layer, output_layer]))
}

# Biases.
biases = {
    'b1': tf.Variable(tf.random_normal([hidden_layer])),
    'out': tf.Variable(tf.random_normal([output_layer]))
}


y_ = model(x, weights, biases)

# Prediction.
pred = tf.nn.softmax(y_)

# Cost.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))

# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Correct prediction.
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))

# Accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        for train_data_input, train_label_input in zip(train_x, train_y):
            _ = sess.run(optimizer, feed_dict = {x: train_data_input, y: train_label_input})

        print("Epoch - ", i + 1)
        #print("Cost - ", cost)
        acc, c = sess.run([accuracy, cost], feed_dict = {x: test_data, y: test_label})
        print("Testing Accuracy - ", acc)
        print("Cost - ", c)