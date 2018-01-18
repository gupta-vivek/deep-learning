# -*- coding: utf-8 -*-
"""
@created on: 17/1/18,
@author: Vivek A Gupta,
@version: v0.0.1

Description:

Sphinx Documentation Status:

..todo::

"""

import read_data
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

train_data, train_label, test_data = read_data.read_data_csv()
train_data = np.asarray(train_data)
train_data = np.reshape(train_data, (train_data.shape[0], 28, 28))

# Time steps
time_steps = 28

# Hidden LSTM units
num_units = 128

# Rows of 28 pixels
n_input = 28

# Learning rate
learning_rate = 0.001

# Output
n_classes = 10

# Batch size
batch_size = 128

# Weight
out_weight = tf.Variable(tf.random_normal([num_units, n_classes]))

# Bias
out_bias = tf.Variable(tf.random_normal([n_classes]))

# Input placeholder
x = tf.placeholder("float", [None, time_steps, n_input])

# Output placeholder
y = tf.placeholder("float", [None, n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

# Network
lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1],out_weight) + out_bias

# Loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Model Evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(50):
        _, los, acc = sess.run([opt, loss, accuracy], feed_dict={x: train_data, y: train_label})
        print("Loss - ", los)
        print("Accuracy - ", acc)
