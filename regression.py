import numpy as np
from numpy.linalg import inv
from numpy.random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time

fontsize = 18
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.loc': 'best', 'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

mpl.style.use('fivethirtyeight')


class regression:
    def __init__(self, x, y, data, noise, n):
        self.x = x
        self.y = y
        self.data = data + noise
        self.noise = noise
        self.n = n

    # Functions helpful for regression
    def create_feature_matrix(self, degree):
        # maybe add jit?
        points = int(degree * (degree + 3) / 2) + 1
        X = np.zeros((self.n, points))

        index = 0
        for i in range(0, degree + 1):
            for j in range(i, degree + 1):
                if (i + j <= degree and index < points):
                    if (i == j):
                        X[:, index] = self.x**(j) * self.y**(i)
                        index += 1
                    else:
                        X[:, index] = self.x**(j) * self.y**(i)
                        index += 1
                        X[:, index] = self.x**(i) * self.y**(j)
                        index += 1
        return X

    def split_data(self, X, ratio=0.2):
        return train_test_split(
            X, self.data, test_size=ratio)

    def scale_data(self, X1, X2):
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

    def create_split_scale(self, degree, ratio=0.2):
        X = self.create_feature_matrix(degree)
        X_train, X_test, y_train, y_test = self.split_data(X, ratio)
        scaled_X_train, scaled_X_test = self.scale_data(X_train, X_test)

        X_split = np.asarray((X_train, X_test))
        y_split = np.asarray((y_train, y_test))
        scaled_X_split = np.asarray((scaled_X_train, scaled_X_test))
        # return X_split, y_split, scaled_X_split
        return scaled_X_train, scaled_X_test, y_train, y_test

    def accuracy_printer(self, train, tilde, test, predict, txt):
        print(txt)
        print("R2 score training:    ", self.R2(train, tilde))
        print("R2 score prediction:  ", self.R2(test, predict))
        print("MSE score training:   ", self.MSE(train, tilde))
        print("MSE score prediciton: ", self.MSE(test, predict), "\n")

    def bootstrap(self, data):
        N = len(data)
        t = np.zeros(N)
        t0 = time()
        for i in range(N):
            t[i] = np.mean(data[randint(0, N - 1, N - 1)])

        print("Runtime: %g sec" % (time() - t0))
        print("Bootstrap Statistics :")
        print("data mean   data std    bootstrap mean   bootstrap std")
        print("%8g    %7g %14g  %15g" % (np.mean(data), np.std(data), np.mean(t), np.std(t)))
        return t

    def ols_beta(self, X, y):
        return inv(X.T @ X) @ X.T @ y

    def ridge_beta(self, X, y, lmb):
        size = len(X[0])
        Id_mtrx = np.eye(size, size)
        return inv(X.T @ X + lmb * Id_mtrx) @ X.T @ y

    def make_prediction(self, X1, X2, beta):
        return (X1 @ beta), (X2 @ beta)

    def lasso(self, X1, X2, y, lmb):
        #X1 - train, X2 - test, y - train
        beta = skl.Lasso(alpha=lmb).fit(X1, y)
        y_fit = beta.predict(X1)
        y_pred = beta.predict(X2)
        return beta, y_fit, y_pred

    def make_MSE_plot(self, max_degree):
        # might make this in a plotter class instead
        test_error = np.zeros(max_degree)
        train_error = np.zeros(max_degree)
        poly_degree = np.arange(0, max_degree)

        for degree in range(0, max_degree):
            scaled_X_train, scaled_X_test, y_train, y_test = self.create_split_scale(degree)
            beta = self.ols_beta(scaled_X_train, y_train)
            y_tilde, y_predict = self.make_prediction(scaled_X_train, scaled_X_test, beta)
            train_error[degree] = self.MSE(y_train, y_tilde)
            test_error[degree] = self.MSE(y_test, y_predict)

        plt.plot(poly_degree, train_error, label="Train Error")
        plt.plot(poly_degree, test_error, label="Test Error")
        plt.legend()
        plt.xlabel("Model complexity")
        plt.ylabel("Prediction Error")
        plt.xlim(0, max_degree - 1)
        plt.title("Mean squared error of training vs testing data")
        plt.show()

    def confidence_interval(self, X, it):
        # mean +/- z*sigma, z = z-value aka 1.96 or 95%
        # sigma = est. std of sample mean
        #sigma = s /sqrt(n)
        # s = std of sample data
        # n = sample size
        beta = self.run_several(X, it)
        confidence_interval = np.zeros((len(beta), 3))

        #beta is sample
        for i in range(len(beta)):
            mean = np.mean(beta[:, i])
            std = np.std(beta[:, i])
            sigma = std / np.sqrt(len(beta[0]))
            lower_bound = mean - sigma * 1.96
            upper_bound = mean + sigma * 1.96
            confidence_interval[i, 0] = lower_bound
            confidence_interval[i, 1] = mean
            confidence_interval[i, 2] = upper_bound

        return confidence_interval

    def run_several(self, X, iterations, method="OLS"):
        beta_aggregate = np.zeros((iterations, len(X[0])))

        for i in range(iterations):
            X_train, X_test, y_train, y_test = self.split_data(X)
            scaled_X_train, scaled_X_test = self.scale_data(X_train, X_test)
            beta = self.ols_beta(scaled_X_train, y_train)
            beta_aggregate[i] = beta

        return beta_aggregate

    def R2(self, data, model):
        return (1.0 - np.sum((data - model))**2 / np.sum((data - np.mean(data))**2))

    def MSE(self, data, model):
        n = np.size(model)
        return (1.0 / n * np.sum((data - model)**2))
