import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from helpers import make_one_hot
import matplotlib.pyplot as plt

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


eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
epochs = 50

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(50), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(X_train, Y_train)

        DNN_scikit[i][j] = dnn

        test_accuracy[i][j] = dnn.score(X_test, Y_test)



# visual representation of grid search
# uses seaborn heatmap, you can also do this with matplotlib imshow
import seaborn as sns

sns.set()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$", fontsize=20)
ax.set_xlabel("$\lambda$", fontsize=20)
plt.show()