import pandas as pd
import tensorflow as tf
import numpy as np


# Prepare data.
def data_prepare(csv_file):
    df = pd.read_csv(csv_file)
    df = df.drop(['PassengerId','Fare', 'Name', 'Ticket', 'Cabin'], axis=1)

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
    df.loc[df['Embarked'] == 'S', 'Embarked'] = 0
    df.loc[df['Embarked'] == 'C', 'Embarked'] = 1
    df.loc[df['Embarked'] == 'Q', 'Embarked'] = 2

    df.loc[df['Sex'] == 'male', 'Sex'] = 0
    df.loc[df['Sex'] == 'female', 'Sex'] = 1

    df['Age'] = df['Age'].fillna(df['Age'].median())

    return df

# One hot vector.
def one_hot_vector(label, size):
    output_label = []
    for i in label:
        a = np.zeros(size, dtype='float')
        a[i] = 1
        output_label.append(a)
    return output_label

# Data preparation.
df_train = data_prepare('train.csv')
df_test = data_prepare('test.csv')

train_y = one_hot_vector(df_train["Survived"], 2)
df_train = df_train.drop(['Survived'], axis = 1)
train_x = df_train.values
test_x = df_test.values

# Parameters.
learning_rate = 0.001
epochs = 10
display_step = 5

# Network size.
input_layer = 6
output_layer = 2

# Input.
x = tf.placeholder(tf.float32, [None, input_layer])
y = tf.placeholder(tf.float32, [None, output_layer])

# Weight
w = tf.Variable(tf.random_normal([input_layer, output_layer]))

# Bias.
b = tf.Variable(tf.random_normal([output_layer]))

# Model.
model = tf.add(tf.matmul(x, w), b)

# Prediction
pred = tf.nn.softmax(model)

# Cost.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
#cost = tf.reduce_sum(tf.pow(pred-y, 2))/(2*891)
# Optimizer.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

init = tf.global_variables_initializer()

#print(train_x)
#print(train_y)


with tf.Session() as sess:
    sess.run(init)

    for i in range(epochs):
        _, c, predict = sess.run([optimizer, cost, model], feed_dict = {x: train_x, y: train_y})
        print("Epoch - ", i + 1)
        print("Cost - ", c)
        print("Prediction - ", predict)