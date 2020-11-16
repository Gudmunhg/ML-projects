import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from time import time
from NN import NeuralNetwork
from helpers import *
import seaborn as sns

#element wise operation *
#matrix multiplication @ or matmul

# ensure the same random numbers appear every time
np.random.seed(89)

plt.rcParams['figure.figsize'] = (12, 12)

def load_data():
	# download MNIST dataset
	digits = datasets.load_digits()

	# define inputs and labels
	inputs = digits.images
	labels = digits.target

	return inputs, labels


def split_data(x, y, ratio):
	# flatten the image
	# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
	if (x.shape[1] != None):
		n = len(x)
		x = x.reshape(len(x), -1)

	# one-liner from scikit-learn library
	X_train, X_test, Y_train, Y_test = train_test_split(
	    x, y, train_size=ratio, test_size=(1 - ratio), random_state=20)

	return X_train, X_test, Y_train, Y_test


inputs, labels = load_data()
X_train, X_test, Y_train, Y_test = split_data(inputs, labels, 0.8)
Y_train_onehot = make_one_hot(Y_train)

n, features = X_train.shape
hidden_layer = 50
categories = 10

"""
Create NN first, then run a grid search to find optimal values of eta and lmbda
Then create second class that trains with the optimal parameters
"""

NN = NeuralNetwork(X_train, Y_train_onehot, features, hidden_layer, categories)
a, lr, lmbda, train_accuracy, test_accuracy = NN.grid_search(X_test, Y_test, Y_train)

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()

#a = 95833... wiht lr = 0.005336699231206312 and lmb = 2.1544346900318865


NN2 = NeuralNetwork(X_train, Y_train_onehot, features, hidden_layer, categories)
lr = 0.005336699231206312
lmbda = 2.1544346900318865
NN2.train(lr, lmbda)

prediction = NN2.make_prediction(X_test)

print("Accuracy score on test set: ", accuracy_score(Y_test, prediction))

X_t = X_test.reshape(-1, 8, 8)
# choose some random images to display
n = 6
indices = np.arange(len(X_test))
random_indices = np.random.choice(indices, size=n)
for i in range(0, n):
    plt.subplot(1, n, i+1)
    plt.axis("off")
    plt.imshow(X_t[random_indices[i]], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Correct: %d, Prediction: %d" % (Y_test[random_indices[i]], prediction[random_indices[i]]))

plt.suptitle("Guesses made by neural network trained on MNIST dataset", fontsize="25")
plt.show()