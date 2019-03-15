# -*- coding: utf-8 -*-
"""
@created on: 2019-02-04,
@author: Vivek A Gupta,

Description:
 Crop Operator in Keras and it's equivalent implementation in Tensorflow using slice operator.

..todo::
"""

import tensorflow as tf


def begin_size(*argv, input_tensor=None):
    begin = [0]
    size = [-1]
    final = [i for i in argv]
    for index, i in enumerate(final):
        begin.append(i[0])
        size.append(int(input_tensor.shape[index + 1]) - i[1] - i[0])
    begin.append(0)
    size.append(-1)

    return begin, size

# 1D
inp_a = tf.constant([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                     [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                     [[7, 7, 7], [8, 8, 8], [9, 9, 9]]])

rank_a = tf.rank(inp_a)

crop_a = [0, 1]
begin_input, size_input = begin_size(crop_a, input_tensor=inp_a)
op_a = tf.keras.layers.Cropping1D(cropping=(crop_a[0], crop_a[1])).call(inputs=inp_a)
slice_op_a = tf.slice(inp_a,begin=tf.constant(begin_input), size=tf.constant(size_input))

# 2D
inp_b = tf.constant([[[[1, 1, 1], [2, 2, 2], [1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4], [3, 3, 3], [4, 4, 4]]],
                     [[[5, 5, 5], [6, 6, 6], [5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8], [7, 7, 7], [8, 8, 8]]]])
                     # [[[9, 9, 9], [10, 10, 10], [9, 9, 9], [10, 10, 10]], [[11, 11, 11], [12, 12, 12], [11, 11, 11], [12, 12, 12]]]])
crop_b1 =[0, 1]
crop_b2 =[1 ,0]
begin_input, end_input = begin_size(crop_b1, crop_b2, input_tensor=inp_b)
op_b = tf.keras.layers.Cropping2D(cropping=((crop_b1[0], crop_b1[1]),(crop_b2[0], crop_b2[1]))).call(inputs=inp_b)
slice_op_b = tf.slice(inp_b,begin=begin_input, size=end_input)

# 3D
inp_c = tf.ones(shape=(3,5,5,5,3))
crop_c1 =[2, 2]
crop_c2 =[2 ,2]
crop_c3 = [2, 2]
begin_input, end_input = begin_size(crop_c1, crop_c2, crop_c3, input_tensor=inp_c)
op_c = tf.keras.layers.Cropping3D(cropping=((crop_c1[0], crop_c1[1]),(crop_c2[0], crop_c2[1]), (crop_c3[0], crop_c3[1]))).call(inputs=inp_c)
slice_op_c = tf.slice(inp_c,begin=begin_input, size=end_input)


with tf.Session() as sess:
    print(sess.run(rank_a))
    print("CROP 1D")
    print(sess.run(inp_a))
    print(inp_a.shape)
    print("\nCrop: ", crop_a)
    op1, op2 = sess.run([op_a, slice_op_a])
    print("\nKeras\n", op1)
    print(op1.shape)
    print("\nTF\n", op2)
    print(op2.shape)

    print("\n\nCROP 2D")
    print(sess.run(inp_b))
    print(inp_b.shape)
    print("\nCrop1: ", crop_b1)
    print("Crop2: ", crop_b2)
    op1, op2 = sess.run([op_b, slice_op_b])
    print("\nKeras\n", op1)
    print(op1.shape)
    print("\nTF\n", op2)
    print(op2.shape)

    print("\n\nCROP 3D")
    print(sess.run(inp_c))
    print(inp_c.shape)
    print("\nCrop1: ", crop_c1)
    print("Crop2: ", crop_c2)
    print("Crop3: ", crop_c3)
    op1, op2 = sess.run([op_c, slice_op_c])
    print("\nKeras\n", op1)
    print(op1.shape)
    print("\nTF\n", op2)
    print(op2.shape)
