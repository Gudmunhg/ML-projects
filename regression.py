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

    def bootstrap(self):
        #Credit Morten Hjort Jensen
        N = self.n
        t = np.zeros(N)
        t0 = time()
        for i in range(N):
            t[i] = np.mean(self.data[randint(0, N - 1, N - 1)])

        print("Runtime: %g sec" % (time() - t0))
        print("Bootstrap Statistics :")
        print("data mean   data std    bootstrap mean   bootstrap std")
        print("%8g    %7g %14g  %15g" % (np.mean(self.data), np.std(self.data), np.mean(t), np.std(t)))
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
        beta = skl.Lasso(alpha=lmb, max_iter=10000).fit(X1, y)
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
        confidence_interval = np.zeros((beta.shape[1], 3))

        #beta is sample
        for i in range(beta.shape[1]):
            mean = np.mean(beta[:, i])
            std = np.std(beta[:, i])
            sigma = std / np.sqrt(beta.shape[0])
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
        return (1.0 - np.sum((data - model)**2) / np.sum((data - np.mean(data))**2))

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

    def bootstrapBiasVariance(self, X_train, y_train, X_test, y_test, n_boot, lmb = 0):
        y_pred = np.zeros((y_test.shape[0], n_boot))
        if lmb == 0:
            #print('OLS')
            for i in range(n_boot):
                # Resample the data n_boot times, making a new prediction for each resampling.
                X_resampled, y_resampled = self.bootstrapResample(X_train, y_train)
                beta_resampled = self.ols_beta(X_resampled, y_resampled)
                y_pred[:,i] = self.make_single_prediction(X_test, beta_resampled)
        else:
            #print('Ridge')
            for i in range(n_boot):
                # Resample the data n_boot times, making a new prediction for each resampling.
                X_resampled, y_resampled = self.bootstrapResample(X_train, y_train)
                beta_resampled = self.ridge_beta(X_resampled, y_resampled, lmb)
                y_pred[:,i] = self.make_single_prediction(X_test, beta_resampled)

        y_test = y_test.reshape(-1,1)
        error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(y_pred, axis=1, keepdims=True))

        print("Error: ", error)
        print("Bias: ", bias)
        print("Variance: ", variance, "\n")

        return error, bias, variance
    """
    def bias_variance(self, y_pred, y_test):
        y_test = y_test.reshape(-1,1)
        error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(y_pred, axis=1, keepdims=True))

        print("Error: ", error)
        print("Bias: ", bias)
        print("Variance: ", variance, "\n")
        return error, bias, variance
    """
    def bias_variance_plot(self, p_start = 1, p_stop = 11, lmb = 0, n_boot = 100):

        p_range = np.arange(p_start,p_stop + 1)
        error = np.zeros(p_range.shape)
        bias = np.zeros(p_range.shape)
        variance = np.zeros(p_range.shape)

        for p in p_range:
            #Create feature matrix
            X = self.create_feature_matrix(p)
            #split data
            X_train, X_test, y_train, y_test = self.split_data(X)
            #Scale data
            scaled_X_train, scaled_X_test = self.scale_data(X_train, X_test)

            ##OLS
            #Create beta
            beta = self.ols_beta(scaled_X_train, y_train)
            #then train the data and get prediction
            y_tilde, y_predict = self.make_prediction(scaled_X_train, scaled_X_test, beta)
            #Write R2/errors out to console
            #test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "OLS scores:")

            print("p = ", p)

            error[p-p_start], bias[p-p_start], variance[p-p_start] = self.bootstrapBiasVariance(scaled_X_train, y_train, scaled_X_test, y_test, n_boot, lmb)

        fig, ax = plt.subplots()

        ax.plot(p_range, error, label = 'Error')
        ax.plot(p_range, bias, label = 'Bias²')
        ax.plot(p_range, variance, label = 'Variance')
        plt.xlabel("Model Complexity $p$")
        plt.ylabel("Error")
        #ax.set_yscale('log')
        plt.legend()
        if lmb == 0:
            plt.title("Bias-Variance Tradeoff OLS")
            fig.savefig("Bias-Variance Tradeoff OLS_n_" + str(self.n))
        else:
            plt.title("Bias-Variance Tradeoff Ridge")
            fig.savefig("Bias-Variance Tradeoff Ridge_n_" + str(self.n))
        plt.show()

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
        lmb = np.logspace(-3, 5, lmb_count)
        MSE_kfold_ridge = np.zeros((lmb_count,splits))
        MSE_kfold_lasso = np.zeros((lmb_count,splits))
        MSE_kfold_ols = np.zeros((lmb_count,splits))

        R2_kfold_ridge = np.zeros((lmb_count,splits))
        R2_kfold_lasso = np.zeros((lmb_count,splits))
        R2_kfold_ols = np.zeros((lmb_count,splits))

        #error_kfold_ols = np.zeros(lmb_count)
        #bias_kfold_ols = np.zeros(lmb_count)
        #variance_kfold_ols = np.zeros(lmb_count)

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
                    R2_kfold_ols[i,j] = self.R2(y_test_kfold,y_pred_kfold_ols)

                beta_kfold_ridge = self.ridge_beta(X_train_kfold, y_train_kfold, lmb[i])
                y_pred_kfold_ridge = self.make_single_prediction(X_test_kfold, beta_kfold_ridge)
                MSE_kfold_ridge[i,j] = self.MSE(y_test_kfold, y_pred_kfold_ridge)
                R2_kfold_ridge[i,j] = self.R2(y_test_kfold,y_pred_kfold_ridge)

                _, _, y_pred_kfold_lasso = self.lasso(X_train_kfold, X_test_kfold, y_train_kfold, lmb[i])
                MSE_kfold_lasso[i,j] = self.MSE(y_test_kfold, y_pred_kfold_lasso)
                R2_kfold_lasso[i,j] = self.R2(y_test_kfold,y_pred_kfold_lasso)

        MSE_kfold_ols = np.mean(MSE_kfold_ols, axis=1)
        MSE_kfold_ols[:] = MSE_kfold_ols[0]
        MSE_kfold_ridge = np.mean(MSE_kfold_ridge, axis=1)
        MSE_kfold_lasso = np.mean(MSE_kfold_lasso, axis=1)

        R2_kfold_ols = np.mean(R2_kfold_ols, axis=1)
        R2_kfold_ols[:] = R2_kfold_ols[0]
        R2_kfold_ridge = np.mean(R2_kfold_ridge, axis=1)
        R2_kfold_lasso = np.mean(R2_kfold_lasso, axis=1)


        fig, ax = plt.subplots()
        ax.plot(lmb, MSE_kfold_ols, label = "Ordinary Least Squares")
        ax.plot(lmb, MSE_kfold_ridge, label = "Ridge Regression")
        ax.plot(lmb, MSE_kfold_lasso, label = "Lasso Regression")

        plt.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')

        plt.xlabel("Hyperparameter $\lambda$")
        plt.ylabel("Estimated MSE")
        plt.title("MSE with k-fold cross validation")
        plt.xlim(lmb[0], lmb[-1]+1)
        plt.ylim(0.8*np.min(np.array((np.min(MSE_kfold_ridge), np.min(MSE_kfold_ols), np.min(MSE_kfold_lasso)))), 1.2*np.max(np.array((np.max(MSE_kfold_ridge), np.max(MSE_kfold_ols), np.max(MSE_kfold_lasso)))))
        plt.show()
        fig.savefig("K-fold-MSE_n_" + str(self.n))

        fig2, ax2 = plt.subplots()
        ax2.plot(lmb, R2_kfold_ols, label = "Ordinary Least Squares")
        ax2.plot(lmb, R2_kfold_ridge, label = "Ridge Regression")
        ax2.plot(lmb, R2_kfold_lasso, label = "Lasso Regression")

        plt.legend()
        ax2.set_xscale('log')

        plt.xlabel("Hyperparameter $\lambda$")
        plt.ylabel("Estimated R2 Score")
        plt.title("R2 Score with k-fold cross validation")
        plt.xlim(lmb[0], lmb[-1]+1)
        plt.ylim(-0.5, 1.1)
        plt.show()
        fig2.savefig("K-fold-R2_n_" + str(self.n))

        return
