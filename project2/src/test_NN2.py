import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
from NN2 import NeuralNetwork
from helpers import *
import seaborn as sns
from sklearn.neural_network import MLPClassifier

# ensure the same random numbers appear every time
np.random.seed(89)

plt.rcParams['figure.figsize'] = (12, 12)

def load_franke():
    #N = 20
    N = 4096
    #x = np.arange(0, 1, 1/N)
    #y = np.arange(0, 1, 1/N)
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    #x, y = np.meshgrid(x, y)

    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    Z = term1 + term2 + term3 + term4
    sigma = 0.1
    noise = np.random.normal(scale = sigma, size=Z.shape)
    Z += noise

    x = np.ravel(x)
    y = np.ravel(y)

    p = 5
    N = len(x)
    l_ = int((p + 1) * (p + 2) / 2)
    X = np.ones((N, l_))
    for i in range(1, p + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x**(i - k)) * (y**k)

    return X, Z


def scale_data(X1, X2):
    # Scale two matricses whose first column both contain only ones
    # X1 is training data, X2 is testing data
    #np.newaxis = None
    if (len(X1[0]) == 1):
        return X1, X2
    else:
        X1 = np.delete(X1, 0, axis=1)
        X2 = np.delete(X2, 0, axis=1)

        scaler = StandardScaler()
        scaler.fit(X1)
        X1 = scaler.transform(X1)
        X2 = scaler.transform(X2)

        X1 = np.concatenate((np.ones(len(X1))[:, None], X1), axis=1)
        X2 = np.concatenate((np.ones(len(X2))[:, None], X2), axis=1)
        return X1, X2


def split_data(x, y, ratio):
    #one-liner from scikit-learn library
    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, train_size=ratio, test_size=(1 - ratio), random_state=20)

    return X_train, X_test, Y_train, Y_test


X, Z = load_franke()
Z = np.ravel(Z)
X_train, X_test, Y_train, Y_test = split_data(X,Z, 0.8)
X_train, X_test = scale_data(X_train, X_test)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)

names = ["Sigmoid", "ReLU", "leakyReLU", "tanh"]
for i, j in enumerate([sigmoid, ReLU, leakyReLU, tanh]):
    NN2 = NeuralNetwork(X_train, Y_train, X_train.shape[1], 5, 1, j)
    test_MSE, test_R2 = NN2.grid_search(X_test, Y_test)
    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_R2, annot=True, ax=ax, cmap="viridis")
    ax.set_title("R2 using %s activation" % names[i], fontsize="40")
    ax.set_ylabel("$\eta$", fontsize=35)
    ax.set_xlabel("$\lambda$", fontsize=35)
    ax.tick_params(labelsize=25)
    sns.set(font_scale=2)
    plt.show()
"""
eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)

# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
epochs = 50

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(50), activation='relu',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(X_train, Y_train)

        DNN_scikit[i][j] = dnn

        test_accuracy[i][j] = dnn.score(X_test, Y_test)


sns.set()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$", fontsize=20)
ax.set_xlabel("$\lambda$", fontsize=20)
plt.show()"""