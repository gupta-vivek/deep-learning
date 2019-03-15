# -*- coding: utf-8 -*-
"""
@created on: 2019-01-29,
@author: Vivek A Gupta,

Description:
    Pad Operator

..todo::
"""
import tensorflow as tf

# Rank 1
a1 = tf.constant([1, 2, 3])
p1 = tf.constant([[1, 1]])
p2 = tf.constant([[2, 2]])
p3 = tf.constant([[3, 3]])
p4 = tf.constant([[4, 4]])

# Rank 2
a2 = tf.constant([[1, 2, 3], [4, 5, 6]])
q1 = tf.constant([[1, 1], [1, 1]])
q2 = tf.constant([[1, 1], [2, 2]])
q3 = tf.constant([[2, 2], [2, 2]])
q4 = tf.constant([[2, 2], [3, 3]])
q5 = tf.constant([[3, 3], [3, 3]])


with tf.Session() as sess:
    print("Rank 1")
    print("Input: ", sess.run(a1))
    print("\nPadding: ", sess.run(p1))
    print("Constant: ", sess.run(tf.pad(a1, p1, "CONSTANT")))
    print("Reflect: ", sess.run(tf.pad(a1, p1, "REFLECT", constant_values=5)))
    print("Symmetric: ", sess.run(tf.pad(a1, p1, "SYMMETRIC")))
    print("\nPadding: ", sess.run(p2))
    print("Constant: ", sess.run(tf.pad(a1, p2, "CONSTANT")))
    print("Reflect: ", sess.run(tf.pad(a1, p2, "REFLECT")))
    print("Symmetric: ", sess.run(tf.pad(a1, p2, "SYMMETRIC")))
    print("\nPadding: ", sess.run(p3))
    print("Constant: ", sess.run(tf.pad(a1, p3, "CONSTANT")))
    print("Reflect: Padding dimension should be less than 3")
    print("Symmetric: ", sess.run(tf.pad(a1, p3, "SYMMETRIC")))
    print("\nPadding: ", sess.run(p4))
    print("Constant: ", sess.run(tf.pad(a1, p4, "CONSTANT")))
    print("Reflect: Padding dimension should be less than 3")
    print("Symmetric: Padding dimension should not be greater than 3")
    print("\n\nRank 2")
    print("Input\n", sess.run(a2))
    print("\nPadding\n", sess.run(q1))
    print("Constant\n", sess.run(tf.pad(a2, q1, "CONSTANT")))
    print("Reflect\n", sess.run(tf.pad(a2, q1, "REFLECT")))
    print("Symmetric\n", sess.run(tf.pad(a2, q1, "SYMMETRIC")))
    print("\nPadding\n", sess.run(q2))
    print("Constant\n", sess.run(tf.pad(a2, q2, "CONSTANT")))
    print("Reflect\n", sess.run(tf.pad(a2, q2, "REFLECT")))
    print("Symmetric\n", sess.run(tf.pad(a2, q2, "SYMMETRIC")))
    print("\nPadding\n", sess.run(q3))
    print("Constant\n", sess.run(tf.pad(a2, q3, "CONSTANT")))
    print("Reflect: Padding dimension should be less than 3")
    print("Symmetric\n", sess.run(tf.pad(a2, q3, "SYMMETRIC")))
    print("\nPadding\n", sess.run(q4))
    print("Constant\n", sess.run(tf.pad(a2, q4, "CONSTANT")))
    print("Reflect: Padding dimension should be less than 3")
    print("Symmetric\n", sess.run(tf.pad(a2, q4, "SYMMETRIC")))
    print("\nPadding\n", sess.run(q5))
    print("Constant\n", sess.run(tf.pad(a2, q5, "CONSTANT")))
    print("Reflect: Padding dimension should be less than 3")
    print("Symmetric: Padding dimension should not be greater than 3")
