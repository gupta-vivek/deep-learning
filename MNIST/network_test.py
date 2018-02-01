import pandas as pd
import numpy as np
import tensorflow as tf


# One hot vector function.
def one_hot_vector(label, output_label, size):
    for i in label:
        a = np.zeros(size, dtype='float')
        a[i] = 1
        output_label.append(a)


# Divide into batches.
def divide_batches(input_batch, output_batch, batch_size):
    for i in range(0, len(input_batch), batch_size):
        output_batch.append(input_batch[i: i + batch_size])


# Read training data from csv.
df = pd.read_csv('mnist_train.csv')
train_label = df.label
df = df.drop(['label'], axis=1)
train_data = df.values
print("Size of training set - ", len(train_data))

# Read testing data from csv.
df = pd.read_csv('mnist_test.csv')
test_label = df.label
df = df.drop(['label'], axis = 1)
test_data = df.values
print("Size of testing set - ", len(test_data))

# Convert labels to one hot vector.
train_label_1 = []
one_hot_vector(train_label, train_label_1, 10)
test_label_1 = []
one_hot_vector(test_label, test_label_1, 10)

# Parameters.
learning_rate = 0.01
epochs = 500
display_step = 5
batch_size = 100

# Divide into batches.
train_data_input = []
train_label_input = []

divide_batches(train_data, train_data_input, batch_size)
divide_batches(train_label_1, train_label_input, batch_size)
train_label_2 = [train_label_1[k:k + batch_size] for k in range(0, len(train_label_1), batch_size)]

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
optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# Correct prediction.
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Accuracy.
#accuracy = tf.reduce_mean(correct_pred)*100
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100

init = tf.initialize_all_variables()

print("Training data")
for i in train_data_input:
    print("Data - ", i)
    print("Shape - ", i.shape)
    break
print("Length - ", len(train_data_input))


print("Training label")
for i in train_label_2:
    print("Data - ", i)
    #print("Shape - ", i.shape)
    break
print("Length - ", len(train_label_2))

print("Testing data")
for i in test_data:
    print("Data - ", i)
    print("Shape - ", i.shape)
    break
print("Length - ", len(test_data))

print("Testing label")
for i in test_label_1:
    print("Data - ", i)
    print("Shape - ", i.shape)
    break
print("Length - ", len(test_label_1))



with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        for train, label in zip(train_data_input, train_label_input):
            _  = sess.run(optimizer, feed_dict = {x: train, y: label})
            cc = sess.run(cost, feed_dict = {x: train, y: label})

        print("Epoch - ", i + 1)
        acc = sess.run(accuracy, feed_dict = {x: test_data, y: test_label_1})
        c = sess.run(cost, feed_dict = {x: test_data, y: test_label_1})
        print("Testing Accuracy - ", acc)
        print("Cost - ", c)
