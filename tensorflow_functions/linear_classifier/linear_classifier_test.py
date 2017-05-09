from Linear_Classifier import LinearClassifier
import file

# Data set.
IRIS_DATA = "iris_training.csv"

# Load datasets.
train_data, train_label, test_data, test_label = file.read_csv(filename=IRIS_DATA, split_ratio=[80, 0, 20],
                                                               delimiter=',', header=True, output_label=True)

# Linear Classifier.
lc = LinearClassifier(dimension=4, n_classes=3)

# Fit the model.
lc.fit(x=train_data, y=train_label, steps=10)

# Calculate accuracy and loss.
acc_and_loss = lc.evaluate(x=test_data, y=test_label)
print("Accuracy - ", acc_and_loss['accuracy'])
print("Loss - ", acc_and_loss['loss'])

# Classify two new flower samples.
sample = [[6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7]]

# Predictions.
predictions = lc.predict(x=sample)
print("Predictions: ", predictions)
