import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def load_data():
    return pd.read_csv(r'_____filePath____', 
                       names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

def prepare_data(dataset):
    class_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    dataset['class'] = dataset['class'].map(class_mapping)
    for feature_name in dataset.columns[:-1]:
        max_value = dataset[feature_name].max()
        min_value = dataset[feature_name].min()
        dataset[feature_name] = (dataset[feature_name] - min_value) / (max_value - min_value)
    mask = np.random.rand(len(dataset)) < 0.8
    train_dataset = dataset[mask]
    test_dataset = dataset[~mask]
    return class_mapping, train_dataset, test_dataset

class MultiClassLogisticRegression:

    def __init__(self, iterations, learning_rate):
        self.iterations = iterations
        self.learning_rate = learning_rate

    # either you can put these as static or private
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def binary_fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        weights = np.zeros(X.shape[1])

        for _ in range(self.iterations):
            z = np.dot(X, weights)
            predictions = self.sigmoid(z)
            errors = y - predictions
            adjustments = self.learning_rate * np.dot(errors, X)
            weights += adjustments
        return weights

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1] + 1))
        for index, label in enumerate(self.classes):
            binary_y = np.where(y == label, 1, 0)
            self.weights[index, :] = self.binary_fit(X, binary_y)

    def scores(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.dot(self.weights, X.T)

    def predict(self, X):
        class_scores = self.scores(X)
        return np.argmax(class_scores, axis=0)

#Data plotting on the refined
def plot_data(classifier, train_X, train_y):
    h = .02 
    x_min, x_max = train_X[:, 0].min() - 1, train_X[:, 0].max() + 1
    y_min, y_max = train_X[:, 1].min() - 1, train_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAFFAA','#00AAFF']))
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=ListedColormap(['#FF0000', '#00FF00','#00AAFF']), edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

data = load_data()
class_mapping, train_data, test_data = prepare_data(data)

#Train Data
train_X = train_data.iloc[:, :2].values
train_y = train_data.iloc[:, -1].values
classifier = MultiClassLogisticRegression(iterations=10000, learning_rate=0.01)
classifier.fit(train_X, train_y)
train_predictions = classifier.predict(train_X)
train_accuracy = np.mean(train_predictions == train_y)
print("Training Data Accuracy: ", train_accuracy)

#Test Data
test_X = test_data.iloc[:, :2].values
test_y = test_data.iloc[:, -1].values
test_predictions = classifier.predict(test_X)
test_accuracy = np.mean(test_predictions == test_y)
print("Test Data Accuracy: ", test_accuracy)

#Decision Function
decision_scores = classifier.scores(test_X)
labels = classifier.classes
decision_score_df = pd.DataFrame(decision_scores, index=labels)
print("Decision Function Scores: ")
print(decision_score_df)
print(decision_score_df.to_string(index=False, header=False))

#Plotting
plot_data(classifier, train_X, train_y)