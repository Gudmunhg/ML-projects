import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
from numba import jit
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('fivethirtyeight')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 3, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2, 'figure.titlesize': fontsize + 5}
plt.rcParams.update(newparams)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

"""
This class provides methods for doing regression analysis using OLS, Ridge,
LASSO, bootstrap and k fold cross validation.
"""


class regression:
    def __init__(self, x, y, data):
        self.x = np.ravel(x) if len(x.shape) > 1 else x
        self.y = np.ravel(y) if len(y.shape) > 1 else y
        self.data = np.ravel(data) if len(data.shape) > 1 else data

    @staticmethod
    def FrankeFunction(x, y):
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) -
                              0.25 * ((9 * y - 2)**2))
        term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
        term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
        return term1 + term2 + term3 + term4

    # Functions helpful for regression analysis
    def create_feature_matrix(self, p):
        N = len(self.x)
        l_ = int((p + 1) * (p + 2) / 2)
        X = np.ones((N, l_))

        for i in range(1, p + 1):
            q = int((i) * (i + 1) / 2)
            for k in range(i + 1):
                X[:, q + k] = (self.x**(i - k)) * (self.y**k)

        return X

    def split_data(self, X, ratio=0.2):
        return train_test_split(
            X, self.data, test_size=ratio)

    def scale_data(self, X1, X2, dummy=True):
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

        return scaled_X_train, scaled_X_test, y_train, y_test

    def accuracy_printer(self, train, tilde, test, predict, txt):
        print(txt)
        print("R2 score training:    ", self.R2(train, tilde))
        print("R2 score prediction:  ", self.R2(test, predict))
        print("MSE score training:   ", self.MSE(train, tilde))
        print("MSE score prediciton: ", self.MSE(test, predict), "\n")

    def ols_beta(self, X, y):
        # Calculate beta using OLS analytical formula
        return inv(X.T @ X) @ X.T @ y

    def ridge_beta(self, X, y, lmb):
        # Calculate beta using Ridge analytical formula
        size = len(X[0])
        Id_mtrx = np.eye(size, size)
        return inv(X.T @ X + lmb * Id_mtrx) @ X.T @ y

    def make_prediction(self, X1, X2, beta):
        # Make two predicitons using e.g. X1 as
        # training data and X2 as testing data
        return (X1 @ beta), (X2 @ beta)

    def lasso(self, X1, X2, y, lmb):
        #X1 - train, X2 - test, y - train
        # Calculate beta using Lasso formula and return fit and
        # prediction data. Note, beta is not an array.
        beta = skl.Lasso(alpha=lmb, max_iter=100000).fit(X1, y)
        y_fit = beta.predict(X1)
        y_pred = beta.predict(X2)
        return beta, y_fit, y_pred

    @jit
    def make_MSE_comparison(self, max_degree):
        # Run several test with differing degrees of approximation using the OLS
        # method and return the results.
        # This method splits and scales the training and testing data.
        test_error = np.zeros(max_degree)
        train_error = np.zeros(max_degree)
        poly_degree = np.arange(0, max_degree)

        for degree in range(0, max_degree):
            scaled_X_train, scaled_X_test, y_train, y_test = self.create_split_scale(
                degree)
            beta = self.ols_beta(scaled_X_train, y_train)
            y_tilde, y_predict = self.make_prediction(
                scaled_X_train, scaled_X_test, beta)
            train_error[degree] = self.MSE(y_train, y_tilde)
            test_error[degree] = self.MSE(y_test, y_predict)

        return poly_degree, train_error, test_error

    def analytic_confindence_interval(self, X, beta, noise):
        # Calculate the analytical variance of beta and use this to
        # find the confidence interval of beta with a 95%(1.96) z-value
        # accuracy.
        var = np.diag(inv(X.T @ X)) * np.std(noise)

        confidence_interval = np.zeros((len(var), 3))
        confidence_interval[:, 0] = beta - np.sqrt(var) * 1.96
        confidence_interval[:, 1] = beta
        confidence_interval[:, 2] = beta + np.sqrt(var) * 1.96

        return confidence_interval

    def R2(self, data, model):
        # Return the R2 score of some data and its corresponding model approx.
        return 1.0 - np.sum((data - model)**2) / np.sum((data - np.mean(data))**2)

    def MSE(self, data, model):
        # Return the MSE score of some data and its corresponding model approx.
        n = np.size(model)
        return ((1.0 / n) * np.sum((data - model)**2))

    @staticmethod
    def make_single_prediction(X, beta):
        # Make two predicitons using e.g. X1 as
        # training data
        return (X @ beta)

    def bootstrapResample(self, x, y):
        # Apply the bootstraping resampling method for a dataset
        # Works by selecting random values form the set and
        # inserting them into a new set. This method does not
        # prohibit one value being resampled more than once.
        inds = np.random.randint(0, x.shape[0], size=x.shape[0])
        x_boot = x[inds]
        y_boot = y[inds]
        return x_boot, y_boot

    @jit
    def bootstrapBiasVariance(self, X_train, y_train, X_test, y_test, n_boot, lmb):
        # Calculate the Bias-Variance trade off
        y_pred = np.zeros((y_test.shape[0], n_boot))
        if lmb == 0:
            for i in range(n_boot):
                # Resample the data n_boot times, making a new prediction for each resampling.
                X_resampled, y_resampled = self.bootstrapResample(
                    X_train, y_train)
                beta_resampled = self.ols_beta(X_resampled, y_resampled)
                y_pred[:, i] = self.make_single_prediction(
                    X_test, beta_resampled)
        else:
            for i in range(n_boot):
                # Resample the data n_boot times, making a new prediction for each resampling.
                X_resampled, y_resampled = self.bootstrapResample(
                    X_train, y_train)
                beta_resampled = self.ridge_beta(X_resampled, y_resampled, lmb)
                y_pred[:, i] = self.make_single_prediction(
                    X_test, beta_resampled)

        y_test = y_test.reshape(-1, 1)
        error = np.mean(np.mean((y_test - y_pred)**2, axis=1, keepdims=True))
        bias = np.mean((y_test - np.mean(y_pred, axis=1, keepdims=True))**2)
        variance = np.mean(np.var(y_pred, axis=1, keepdims=True))

        print("Error: ", error)
        print("Bias: ", bias)
        print("Variance: ", variance, "\n")

        return error, bias, variance

    def bias_variance_plot(self, p_start=1, p_stop=11, lmb=0, n_boot=100):

        p_range = np.arange(p_start, p_stop + 1)
        error = np.zeros(p_range.shape)
        bias = np.zeros(p_range.shape)
        variance = np.zeros(p_range.shape)

        for p in p_range:
            X = self.create_feature_matrix(p)
            X_train, X_test, y_train, y_test = self.split_data(X)
            scaled_X_train, scaled_X_test = self.scale_data(X_train, X_test)

            # OLS
            beta = self.ols_beta(scaled_X_train, y_train)
            y_tilde, y_predict = self.make_prediction(
                scaled_X_train, scaled_X_test, beta)

            print("p = ", p)

            error[p - p_start], bias[p - p_start], variance[p - p_start] = self.bootstrapBiasVariance(
                scaled_X_train, y_train, scaled_X_test, y_test, n_boot, lmb)

        fig, ax = plt.subplots()

        ax.plot(p_range, error, label='Error')
        ax.plot(p_range, bias, label='Bias²')
        ax.plot(p_range, variance, label='Variance')
        plt.xlabel("Model Complexity $p$")
        plt.ylabel("Error")
        plt.legend()
        if lmb == 0:
            plt.title("Bias-Variance Tradeoff OLS")

        else:
            plt.title(
                r"Bias-Variance Tradeoff Ridge with $\lambda$ = " + str(lmb))
        plt.show()

    def k_fold(self, x, splits=5, shuffle=False):

        indices = np.arange(x.shape[0])
        if shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(indices)

        test_inds = np.array_split(indices, splits)
        train_inds = np.array_split(indices, splits)
        for i in range(splits):
            train_inds[i] = np.concatenate(np.delete(test_inds, i, 0))

        return test_inds, train_inds

    def cross_validation(self, X, y, splits):

        test_inds, train_inds = self.k_fold(X, splits)
        lmb_count = 100
        lmb = np.logspace(-3, 5, lmb_count)
        MSE_kfold_ridge = np.zeros((lmb_count, splits))
        MSE_kfold_lasso = np.zeros((lmb_count, splits))
        MSE_kfold_ols = np.zeros((lmb_count, splits))

        R2_kfold_ridge = np.zeros((lmb_count, splits))
        R2_kfold_lasso = np.zeros((lmb_count, splits))
        R2_kfold_ols = np.zeros((lmb_count, splits))

        for i in range(lmb_count):
            for j in range(splits):
                X_train_kfold = X[train_inds[j]]
                y_train_kfold = y[train_inds[j]]

                X_test_kfold = X[test_inds[j]]
                y_test_kfold = y[test_inds[j]]

                if i == 0:
                    beta_kfold_ols = self.ols_beta(
                        X_train_kfold, y_train_kfold)
                    y_pred_kfold_ols = self.make_single_prediction(
                        X_test_kfold, beta_kfold_ols)
                    MSE_kfold_ols[i, j] = self.MSE(
                        y_test_kfold, y_pred_kfold_ols)
                    R2_kfold_ols[i, j] = self.R2(
                        y_test_kfold, y_pred_kfold_ols)

                beta_kfold_ridge = self.ridge_beta(
                    X_train_kfold, y_train_kfold, lmb[i])
                y_pred_kfold_ridge = self.make_single_prediction(
                    X_test_kfold, beta_kfold_ridge)
                MSE_kfold_ridge[i, j] = self.MSE(
                    y_test_kfold, y_pred_kfold_ridge)
                R2_kfold_ridge[i, j] = self.R2(
                    y_test_kfold, y_pred_kfold_ridge)

                _, _, y_pred_kfold_lasso = self.lasso(
                    X_train_kfold, X_test_kfold, y_train_kfold, lmb[i])
                MSE_kfold_lasso[i, j] = self.MSE(
                    y_test_kfold, y_pred_kfold_lasso)
                R2_kfold_lasso[i, j] = self.R2(
                    y_test_kfold, y_pred_kfold_lasso)

        MSE_kfold_ols = np.mean(MSE_kfold_ols, axis=1)
        MSE_kfold_ols[:] = MSE_kfold_ols[0]
        MSE_kfold_ridge = np.mean(MSE_kfold_ridge, axis=1)
        MSE_kfold_lasso = np.mean(MSE_kfold_lasso, axis=1)

        R2_kfold_ols = np.mean(R2_kfold_ols, axis=1)
        R2_kfold_ols[:] = R2_kfold_ols[0]
        R2_kfold_ridge = np.mean(R2_kfold_ridge, axis=1)
        R2_kfold_lasso = np.mean(R2_kfold_lasso, axis=1)

        fig, ax = plt.subplots()
        ax.plot(lmb, MSE_kfold_ols, label="Ordinary Least Squares")
        ax.plot(lmb, MSE_kfold_ridge, label="Ridge Regression")
        ax.plot(lmb, MSE_kfold_lasso, label="Lasso Regression")

        plt.legend()
        ax.set_xscale('log')

        plt.xlabel(r"Hyperparameter $\lambda$")
        plt.ylabel("Estimated MSE")
        plt.title("MSE with k-fold cross validation")
        plt.xlim(lmb[0], lmb[-1] + 1)
        plt.ylim(0.8 * np.min(np.array((np.min(MSE_kfold_ridge), np.min(MSE_kfold_ols), np.min(MSE_kfold_lasso)))),
                 1.2 * np.max(np.array((np.max(MSE_kfold_ridge), np.max(MSE_kfold_ols), np.max(MSE_kfold_lasso)))))
        plt.show()

        fig2, ax2 = plt.subplots()
        ax2.plot(lmb, R2_kfold_ols, label="Ordinary Least Squares")
        ax2.plot(lmb, R2_kfold_ridge, label="Ridge Regression")
        ax2.plot(lmb, R2_kfold_lasso, label="Lasso Regression")

        plt.legend()
        ax2.set_xscale('log')

        plt.xlabel(r"Hyperparameter $\lambda$")
        plt.ylabel("Estimated R2 Score")
        plt.title("R2 Score with k-fold cross validation")
        plt.xlim(lmb[0], lmb[-1] + 1)
        plt.ylim(-0.5, 1.1)
        plt.show()

        return
