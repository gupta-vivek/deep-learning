import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Learning rate.
learning_rate = 0.001
epoch = 10
display_step = 10

# Training data.
# Input data.
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# Output data.
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

# Testing data.
# Input data.
test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
print(test_X)
# Output data.
test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

# Size of the input.
n = len(train_X)

# Bias.
b = tf.Variable(np.random.randn(), name = 'bias')
# Weight.
w = tf.Variable(np.random.randn(), name = 'weight')

# Input.
x = tf.placeholder("float")
# Output.
y = tf.placeholder("float")

# Prediction.
pred = tf.add(tf.mul(x, w), b)

# Cost function.
# Mean squared error.
cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*n)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables.
#init = tf.global_variables_initializer()
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        for (j, k) in zip(train_X, train_Y):
            print(j)
            sess.run(optimizer, feed_dict = {x: train_X, y: train_Y})

        if i % display_step == 0:
            print("Epoch - ", i)
            print("Cost = ", sess.run(cost, feed_dict = {x: train_X, y: train_Y}))
            print("Weight = " , sess.run(w))
            print("Bias = ", sess.run(b))

    print("Optimization Finished.")
    c = sess.run(cost, feed_dict = {x: train_X, y: train_Y})
    print("Cost - ", c)
    print("Weight = ", sess.run(w))
    print("Bias = ", sess.run(b))

    # Graph.
    """
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
    """

    print("Testing")
    for i, j in zip(test_X, test_Y):
        p = sess.run(pred, feed_dict = {x: i, y: j})
        print("Prediction: ", p, "Correct value: ", j)


    # Graph.
    plt.plot(test_X, test_Y, 'ro', label = "Testing data")
    plt.plot(train_X, sess.run(w)*train_X + sess.run(b), label="Fitted line")
    plt.legend()
    plt.show()
