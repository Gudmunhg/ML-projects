import numpy as np
from numpy.linalg import inv
from numpy.random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time


mpl.style.use('fivethirtyeight')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 5, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15,10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best', 'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

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

        #X_split = np.asarray((X_train, X_test))
        #y_split = np.asarray((y_train, y_test))
        #scaled_X_split = np.asarray((scaled_X_train, scaled_X_test))
        # return X_split, y_split, scaled_X_split
        return scaled_X_train, scaled_X_test, y_train, y_test

    def accuracy_printer(self, train, tilde, test, predict, txt):
        print(txt)
        print("R2 score training:    ", self.R2(train, tilde))
        print("R2 score prediction:  ", self.R2(test, predict))
        print("MSE score training:   ", self.MSE(train, tilde))
        print("MSE score prediciton: ", self.MSE(test, predict), "\n")

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
        beta = skl.Lasso(alpha=lmb, max_iter=100000).fit(X1, y)
        y_fit = beta.predict(X1)
        y_pred = beta.predict(X2)
        return beta, y_fit, y_pred

    def make_MSE_plot(self, max_degree):
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

    def run_several(self, X, iterations, method="OLS"):
        beta_aggregate = np.zeros((iterations, len(X[0])))

        for i in range(iterations):
            X_train, X_test, y_train, y_test = self.split_data(X)
            scaled_X_train, scaled_X_test = self.scale_data(X_train, X_test)
            beta = self.ols_beta(scaled_X_train, y_train)
            beta_aggregate[i] = beta

        return beta_aggregate

    def analytic_confindence_interval(self, X, beta):
        var = np.diag(inv(X.T @ X)) * np.std(self.noise)

        confidence_interval = np.zeros((len(var), 3))
        confidence_interval[:, 0] = beta - np.sqrt(var) * 1.96 
        confidence_interval[:, 1] = beta
        confidence_interval[:, 2] = beta + np.sqrt(var) * 1.96 

        return confidence_interval

    def R2(self, data, model):
        return 1.0 - np.sum((data - model)**2) / np.sum((data - np.mean(data))**2)

    def MSE(self, data, model):
        n = np.size(model)
        return (1.0 / n * np.sum((data - model)**2))

    def make_single_prediction(self, X, beta):
        return (X @ beta)

    def bootstrapResample(self, x, y):
        inds = np.random.randint(0, x.shape[0], size = x.shape[0])
        x_boot = x[inds]
        y_boot = y[inds]
        return x_boot, y_boot

    def bootstrapBiasVariance(self, X_train, y_train, X_test, y_test, n_boot):
        y_pred = np.zeros((y_test.shape[0], n_boot))

        for i in range(n_boot):
            # Resample the data n_boot times, making a new prediction for each resampling.
            X_resampled, y_resampled = self.bootstrapResample(X_train, y_train)
            beta_resampled = self.ols_beta(X_resampled, y_resampled)
            y_pred[:,i] = self.make_single_prediction(X_test, beta_resampled).ravel()

        #y_test_ = np.zeros(y_pred.shape)
        #for i in range(n_boot):
        #    y_test_[:,i] = y_test
        error = np.mean( np.mean((y_test[:,np.newaxis] - y_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance = np.mean( np.var(y_pred, axis=1, keepdims=True))

        print("Error: ", error)
        print("Bias: ", bias)
        print("Variance: ", variance)
        return

    def k_fold(self, x, splits = 5, shuffle = False):

        indices = np.arange(x.shape[0])
        if shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(indices)

        test_inds = np.array_split(indices, splits)
        train_inds = np.array_split(indices, splits)
        for i in range(splits):
            train_inds[i] = np.concatenate(np.delete(test_inds, i, 0))

        return test_inds, train_inds

    def ridge_cross_validation(self, X, y, splits):

        test_inds, train_inds = self.k_fold(X, splits)
        lmb_count = 500
        lmb = np.logspace(-3, 3, lmb_count)
        MSE_kfold_ridge = np.zeros((lmb_count,splits))
        MSE_kfold_lasso = np.zeros((lmb_count,splits))
        MSE_kfold_ols = np.zeros((lmb_count,splits))

        for i in range(lmb_count):
            for j in range(splits):
                X_train_kfold = X[train_inds[j]]
                y_train_kfold = y[train_inds[j]]

                X_test_kfold = X[test_inds[j]]
                y_test_kfold = y[test_inds[j]]

                if i == 0:
                    beta_kfold_ols = self.ols_beta(X_train_kfold, y_train_kfold)
                    y_pred_kfold_ols = self.make_single_prediction(X_test_kfold, beta_kfold_ols)
                    MSE_kfold_ols[i,j] = self.MSE(y_test_kfold, y_pred_kfold_ols)

                beta_kfold_ridge = self.ridge_beta(X_train_kfold, y_train_kfold, lmb[i])
                y_pred_kfold_ridge = self.make_single_prediction(X_test_kfold, beta_kfold_ridge)
                MSE_kfold_ridge[i,j] = self.MSE(y_test_kfold, y_pred_kfold_ridge)

                _, _, y_pred_kfold_lasso = self.lasso(X_train_kfold, X_test_kfold, y_train_kfold, lmb[i])
                """
                beta_kfold_lasso = skl.Lasso(alpha=lmb[i]).fit(X_train_kfold, y_train_kfold)
                y_pred_kfold_lasso = self.make_single_prediction(X_test_kfold, beta_kfold_lasso)
                """
                MSE_kfold_lasso[i,j] = self.MSE(y_test_kfold, y_pred_kfold_lasso)

        MSE_kfold_ols = np.mean(MSE_kfold_ols, axis=1)
        MSE_kfold_ols[:] = MSE_kfold_ols[0]
        MSE_kfold_ridge = np.mean(MSE_kfold_ridge, axis=1)
        MSE_kfold_lasso = np.mean(MSE_kfold_lasso, axis=1)

        fig, ax = plt.subplots()
        ax.plot(lmb, MSE_kfold_ols, label = "Ordinary Least Squares")
        ax.plot(lmb, MSE_kfold_ridge, label = "Ridge Regression")
        ax.plot(lmb, MSE_kfold_lasso, label = "Lasso Regression")

        plt.legend()
        #ax.set_yscale('log')
        ax.set_xscale('log')
        plt.xlabel("Hyperparameter $\lambda$")
        plt.ylabel("Estimated MSE")
        plt.title("MSE k-fold cross validation")
        plt.xlim(lmb[0], lmb[-1]+1)
        plt.show()
        fig.savefig("K-fold-MSE")

        return
