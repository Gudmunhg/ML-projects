from structs import hidden_neurons, output_layer
import numpy as np
from helpers import sigmoid, softmax, accuracy_score

class NeuralNetwork:
    def __init__(self, X, Y, features, hidden_layers, categories, activation: object):
        #Assume X, Y is X_train and Y_train_oneshot
        self.X = X
        self.Y = Y
        np.random.seed(89)
        self.hidden = hidden_neurons(hidden_layers, features)
        self.output = output_layer(hidden_layers, categories)
        self.act = activation
        self.activation = activation()

    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        # weighted sum of inputs to the hidden layer for
        # each input image and each hidden neuron.
        XW_h = X @ self.hidden.weights
        Z_h = XW_h + self.hidden.bias

        # Activation in the hidden layer
        A_h = self.activation(Z_h)

        # weighted sum of inputs to the output layer for
        # each input image and each hidden neuron.
        Z_o = (A_h @ self.output.weights) + self.output.bias

        self.probabilities = softmax(Z_o)

        """if (np.isnan(self.probabilities).any()):
            print("Nan encountered!")
            exit()"""

        # unit test
        # Check that the values of probabilities sum up to 1.0
        for i in range(self.probabilities.shape[0]):
            prob_sum = self.probabilities[i].sum()
            if prob_sum < 1.0 + 1e-16 and prob_sum > 1.0 - 1e-16:
                print(self.probabilities[i].sum())
                print("Error! Probabilities do not sum up to 1.0!")

        return self.probabilities, A_h

    def back_propagation(self, X, Y, eta, lmbda):
        probabilities, A_h = self.feed_forward(X)

        # Output layer error:
        #this is based on using sigmoid/softmax as activation, with cross
        #entropy as cost function
        output_error = probabilities - Y

        # Hidden layer error
        #Add derivative of last layer activation as general function
        hidden_error = (output_error @ self.output.weights.T) * A_h * (1 - A_h)

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
                #datapoints = np.random.randint(1, num_batches)
                datapoints = np.random.choice(indices, size=batch_size, replace=False)

                X_data = self.X[datapoints]
                Y_data = self.Y[datapoints]

                self.back_propagation(X_data, Y_data, eta, lmbda)

    def make_prediction(self, X):
        probabilities, _ = self.feed_forward(X)
        return np.argmax(probabilities, axis=1)

    def grid_search(self, X_test, Y_test, Y_train):
        eta_vals = np.logspace(-5, 1, 7)
        lmbda_vals = np.logspace(-5, 1, 7)

        #results = np.zeros((len(eta_vals), len(lmbda_vals)))
        accuracy = 0
        lmb, lr = 0, 0
        DNN_numpy = np.zeros((len(eta_vals), len(lmbda_vals)), dtype=object)
        train_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))
        test_accuracy = np.zeros((len(eta_vals), len(lmbda_vals)))

    #dataclasses are not reset each time, so results become wrong
        for i, eta in enumerate(eta_vals):
            for j, lmbda in enumerate(lmbda_vals):
                NN = NeuralNetwork(self.X, self.Y, self.hidden.features, self.hidden.n, self.output.categories, self.act)

                NN.train(eta, lmbda)
                DNN_numpy[i][j] = NN

                train_pred = NN.make_prediction(self.X)
                predict = NN.make_prediction(X_test)

                train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
                test_accuracy[i][j] = accuracy_score(Y_test, predict)

                score = accuracy_score(Y_test, predict)

                if (score > accuracy):
                    accuracy = score
                    lr = eta
                    lmb = lmbda

        return accuracy, lr, lmb, train_accuracy, test_accuracy
