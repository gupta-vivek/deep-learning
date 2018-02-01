# -*- coding: utf-8 -*-
"""
@created on: 22/1/18,
@author: Vivek A Gupta,
@version: v0.0.1

Description:

Sphinx Documentation Status:

..todo::

"""

import tensorflow as tf
import read_data
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

batch_size = 100

train_data, train_label, test_data = read_data.read_data_csv()

train_data = read_data.divide_batches(train_data, batch_size)


def sample_noise(m, n):
    return np.random.uniform(-1., 1., size = [m, n])


x = tf.placeholder(tf.float32, shape=[None, 784])
# Discriminator
disc_weight = tf.Variable(tf.random_normal([784, 128]))
disc_bias = tf.Variable(tf.zeros([128]))
disc_weight_out = tf.Variable(tf.random_normal([128, 1]))
disc_bias_out = tf.Variable(tf.random_normal([1]))

disc_vars = [disc_weight, disc_bias, disc_weight_out, disc_bias_out]


z = tf.placeholder(tf.float32, shape=[None, 100])
# Generator
gen_weight = tf.Variable(tf.random_normal([100, 128]))
gen_bias = tf.Variable(tf.zeros([128]))
gen_weight_out = tf.Variable(tf.random_normal([128, 784]))
gen_bias_out = tf.Variable(tf.zeros([784]))

gen_vars = [gen_weight, gen_bias, gen_weight_out, gen_bias_out]


def discriminator(x):
    disc_hidden = tf.nn.relu(tf.add(tf.matmul(x, disc_weight), disc_bias))
    disc_out = tf.add(tf.matmul(disc_hidden, disc_weight_out), disc_bias_out)

    return disc_out


def generator(z):
    gen_hidden = tf.nn.relu(tf.add(tf.matmul(z, gen_weight), gen_bias))
    gen_out = tf.add(tf.matmul(gen_hidden, gen_weight_out), gen_bias_out)

    return tf.nn.sigmoid(gen_out)


# Learning rate
learning_rate = 0.002

disc_real = discriminator(x)
disc_fake = generator(z)
gen_sample = disc_fake

# Loss
disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))

disc_loss = disc_fake_loss + disc_real_loss

gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_sample, labels=tf.ones_like(gen_sample)))

# Optimizer
disc_opt = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=disc_vars)
gen_opt = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=gen_vars)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(100):
        for data in train_data:
            batch_z = np.random.uniform(-1, 1, size=(100, 100))
            _, d_loss = sess.run([disc_opt, disc_loss], feed_dict={x: data, z: batch_z})
            _, g_loss = sess.run([gen_opt, gen_loss], feed_dict={z: batch_z})

        print("\nEpoch - ", i)
        print("Discriminator Loss - ", d_loss)
        print("Generator Loss - ", g_loss)

        batch_z = np.random.uniform(-1, 1, size=(100, 100))
        sample_output = sess.run(gen_sample, feed_dict={z: batch_z})

        for ind, image in enumerate(sample_output):
            image = image.reshape([28, 28]).astype('uint8')*255
            img = Image.fromarray(image)
            img.save('image/' + str(ind) + '.png')
