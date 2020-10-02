import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl


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

        if (degree == 0):
            X[:, 0] = 1.0
        else:
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

    def split_data(self, X, data, ratio=0.2):
        return train_test_split(
            X, data, test_size=ratio, random_state=42)

    def scale_data(self, X1, X2):
        # Scale two matricses whose first column both contain only ones
        # X1 is training data, X2 is testing data
        #np.newaxis = None

        if (len(X1) == 1):
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

    def accuracy_printer(self, train, tilde, test, predict, txt):
        print(txt)
        print("R2 score training:    ", self.R2(train, tilde))
        print("R2 score prediction:  ", self.R2(test, predict))
        print("MSE score training:   ", self.MSE(train, tilde))
        print("MSE score prediciton: ", self.MSE(test, predict), "\n")

    def bootstrap(self):
        pass

    def ols_beta(self, X, y):
        return inv(X.T @ X) @ X.T @ y

    def ridge(self, X, y, lmb):
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

    def R2(self, data, model):
        return (1.0 - np.sum((data - model))**2 / np.sum((data - np.mean(data))**2))

    def MSE(self, data, model):
        n = np.size(model)
        return (1.0 / n * np.sum((data - model)**2))
