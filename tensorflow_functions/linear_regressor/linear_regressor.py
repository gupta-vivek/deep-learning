import tensorflow as tf
from tensorflow.contrib.learn import LinearRegressor
import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Input data.
train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
# Output data.
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
# Result
result = [1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3]

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]


# Linear regressor.
linear_regressor = LinearRegressor(feature_columns=feature_columns, model_dir="/tmp/linear_regressor")


# Define the training inputs
def get_train_inputs():
    x = tf.constant(train_X)
    y = tf.constant(train_Y)
    return x, y

# Fit model.
# linear_regressor.fit(input_fn=get_train_inputs, steps=5)

# Define the test inputs
def get_test_inputs():
    x = tf.constant(train_X)
    return x

# Evaluate loss.
loss = linear_regressor.evaluate(input_fn=get_train_inputs, steps=1)['loss']
print("Loss - ", loss)

# Predictions.
predictions = list(linear_regressor.predict(input_fn=get_test_inputs))
print("New Samples, Class Predictions:    {}\n".format(predictions))

# Accuracy.
count = 0
for i, j in zip(predictions, result):
    if (j - 0.5) < i < (j + 0.5):
        count = count + 1

accuracy = count/len(result) * 100

print("Accuracy - ", accuracy)
