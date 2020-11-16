from structs import hidden_neurons, output_layer
import numpy as np
from helpers import *

class NeuralNetwork:
    def __init__(self, X, Y, features, hidden_layers, categories, activation: object):
        self.X = X
        self.Y = Y
        np.random.seed(89)
        self.hidden = hidden_neurons(hidden_layers, features)
        self.output = output_layer(hidden_layers, categories)
        self.activation = activation()
        self.act = activation

    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        # weighted sum of inputs to the hidden layer for
        # each input image and each hidden neuron.
        XW_h = X @ self.hidden.weights
        Z_h = XW_h + self.hidden.bias

        # Activation in the hidden layer
        A_h = self.activation(Z_h)
        self.der = self.activation.derivative(Z_h)

        # weighted sum of inputs to the output layer for
        # each input image and each hidden neuron.
        Z_o = (A_h @ self.output.weights) + self.output.bias
        self.der2 = self.activation.derivative(Z_o)

        self.probabilities = Z_o

        return self.probabilities, A_h

    def back_propagation(self, X, Y, eta, lmbda):
        probabilities, A_h = self.feed_forward(X)

        # Output layer error:
        output_error = (probabilities - Y.reshape(-1,1)) * self.der2

        # Hidden layer error
        #Add derivative of last layer activation as general function
        hidden_error = (output_error @ self.output.weights.T) * self.der

        # gradients
        output_weight_gradient = A_h.T @ output_error
        output_bias_gradient = np.sum(output_error, axis=0)

        hidden_weight_gradient = X.T @ hidden_error
        hidden_bias_gradient = np.sum(hidden_error, axis=0)

        #regularization term gradients
        if (lmbda > 0.0):
            output_weight_gradient += lmbda * self.output.weights
            hidden_weight_gradient += lmbda * self.hidden.weights

        self.output.weights -= eta * output_weight_gradient
        self.output.bias -= eta * output_bias_gradient

        self.hidden.weights -= eta * hidden_weight_gradient
        self.hidden.bias -= eta * hidden_bias_gradient

        return output_weight_gradient, output_bias_gradient, hidden_weight_gradient, hidden_bias_gradient

    def train(self, eta, lmbda):
        indices = np.arange(self.X.shape[0])
        epochs = 10
        batch_size = 100
        iterations = self.X.shape[0] // batch_size

        #Replace with SGD
        for i in range(epochs):
            for j in range(iterations):
                datapoints = np.random.choice(indices, size=batch_size, replace=False)

                X_data = self.X[datapoints]
                Y_data = self.Y[datapoints]

                self.back_propagation(X_data, Y_data, eta, lmbda)

    def make_prediction(self, X):
        probabilities, _ = self.feed_forward(X)
        return probabilities

    def MSE(self, Y, prediction):
        N = np.size(Y)
        return np.sum((Y - prediction)**2)/N

    def R2(self, Y, prediction):
        return (1.0 - np.sum((Y - prediction)**2) / np.sum((Y - np.mean(Y))**2))

    def grid_search(self, X_test, Y_test):
        eta_vals = np.logspace(-5, -2, 4)
        lmbda_vals = np.logspace(-5, 1, 7)

        accuracy = pow(10,10)
        lmb, lr = 0, 0

        DNN_numpy = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)
        test_MSE = np.zeros((len(eta_vals), len(lmbda_vals)))
        test_R2 = np.zeros((len(eta_vals), len(lmbda_vals)))

    #dataclasses are not reset each time, so results become wrong
        for i, eta in enumerate(eta_vals):
            for j, lmbda in enumerate(lmbda_vals):
                NN = NeuralNetwork(self.X, self.Y, self.hidden.features, self.hidden.n, self.output.categories, self.act)
                NN.train(eta, lmbda)


                DNN_numpy[i][j] = NN

                predict = NN.make_prediction(X_test)
                test_MSE[i][j] = self.MSE(Y_test, predict)
                test_R2[i][j] = self.R2(Y_test, predict)

                score = self.MSE(Y_test, predict)

                if (score < accuracy):
                    accuracy = score
                    lr = eta
                    lmb = lmbda
        return test_MSE, test_R2
