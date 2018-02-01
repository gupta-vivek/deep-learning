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

train_yv = one_hot_vector(df_train["Survived"], 2)
df_train = df_train.drop(['Survived'], axis = 1)
train_xv = df_train.values

test_x = df_test.values

share = 700
train_x = train_xv[:share]
train_y = train_yv[:share]
valid_x = train_xv[share:]
valid_y = train_yv[share:]

print("Train X - ", len(train_x))
print("Valid X - ", len(valid_x))



# Parameters.
learning_rate = 0.01
epochs = 200
display_step = 5
beta = 0.001

# Network size.
input_layer = 6
hidden_layer_1 = 100
hidden_layer_2 = 100
hidden_layer_3 = 50
output_layer = 2

# Input.
x = tf.placeholder(tf.float32, [None, input_layer])
y = tf.placeholder(tf.float32, [None, output_layer])

# Model.
def model(x, weights, biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['w1']), biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['w2']), biases['b2']))
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['w3']), biases['b3']))
    out = tf.add(tf.matmul(layer3, weights['w4']), biases['b4'])
    #out = tf.add(tf.matmul(layer2, weights['w3']), biases['b3'])
    return out

weights = {
    'w1': tf.Variable(tf.random_normal([input_layer, hidden_layer_1])),
    'w2': tf.Variable(tf.random_normal([hidden_layer_1, hidden_layer_2])),
    'w3': tf.Variable(tf.random_normal([hidden_layer_2, hidden_layer_3])),
    'w4': tf.Variable(tf.random_normal([hidden_layer_3, output_layer]))
    #'w3': tf.Variable(tf.random_normal([hidden_layer_2, output_layer]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([hidden_layer_1])),
    'b2': tf.Variable(tf.random_normal([hidden_layer_2])),
    'b3': tf.Variable(tf.random_normal([hidden_layer_3])),
    'b4': tf.Variable(tf.random_normal([output_layer]))
    #'b3': tf.Variable(tf.random_normal([output_layer]))
}

y_ = model(x, weights, biases)

# Prediction.
pred = tf.nn.softmax(y_)

# Correct prediction.
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# Accuracy.
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) * 100
"""
l2loss = tf.nn.l2_loss(weights['w1']) * beta + tf.nn.l2_loss(weights['w2']) * beta + tf.nn.l2_loss(weights['w3']) * beta\
         + tf.nn.l2_loss(weights['w4']) + tf.nn.l2_loss(biases['b1']) * beta + tf.nn.l2_loss(biases['b2']) * beta\
         + tf.nn.l2_loss(biases['b3'])  + tf.nn.l2_loss(biases['b4'])
"""
l2loss = tf.nn.l2_loss(weights['w1']) + tf.nn.l2_loss(weights['w2']) + tf.nn.l2_loss(weights['w3'])\
         + tf.nn.l2_loss(weights['w4']) + tf.nn.l2_loss(biases['b1']) + tf.nn.l2_loss(biases['b2']) \
         + tf.nn.l2_loss(biases['b3']) + tf.nn.l2_loss(biases['b4'])

# Cost.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y)) + l2loss * beta
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
        _, c = sess.run([optimizer, cost], feed_dict = {x: train_x, y: train_y})
        print("Epoch - ", i + 1)
        print("Cost - ", c)

        acc = sess.run(accuracy, feed_dict = {x: valid_x, y: valid_y})
        print("Accuracy - ", acc)