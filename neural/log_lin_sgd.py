import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from numpy.linalg import inv
from numpy.random import randint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as skl
import matplotlib as mpl
from time import time

class gradient_descent:
    def __init__(self, X, y, minibatch_size, n_epochs = 50, max_iter = 50, t0 = 0.1, t1 = 50):
        self.rng = np.random.default_rng()
        self.X = X #design matrix
        self.y = y #data points
        self.minibatch_size = minibatch_size #size of each minibatch

        self.classes = self.y.shape[1]
        if self.classes > 1:
            self.classes -= 1
        self.complexity = self.X.shape[1]


        self.t0 = t0
        self.t1 = t1 #controlling the learning rate
        self.n_epochs = n_epochs
        self.max_iter = max_iter

        self.data_count = np.size(y, axis=0)
        self.minibatch_count = self.data_count/minibatch_size

    def learning_schedule(self, t):
        return (self.t0 / (1 + t/self.t1) ) / self.minibatch_count

    def sgd(self, errorfunc, gradient):

        tol = 0.0001
        sgd_error = pow(10,10)
        theta_best = np.zeros((self.complexity,self.classes))
        for iter in range(self.max_iter):

            theta = np.random.randn(self.complexity,self.classes)

            for epoch in range(self.n_epochs):
                i = 0
                shuffle = self.rng.choice(self.data_count, size=self.data_count, replace=False) #shuffling the data
                self.X = self.X[shuffle]
                self.y = self.y[shuffle]
                model_batches = [self.X[k:k + self.minibatch_size] for k in np.arange(0, self.data_count, self.minibatch_size)]
                data_batches = [self.y[k:k + self.minibatch_size] for k in np.arange(0, self.data_count, self.minibatch_size)]
                for (model_batch, data_batch) in zip(model_batches, data_batches):
                    eta = self.learning_schedule((epoch*self.minibatch_count + i))
                    i += 1
                    #print("error", errorfunc(self.X, self.y, theta))
                    #print(model_batch.shape)
                    #print(data_batch.shape)
                    #print(theta.shape)
                    #print("gradient", np.max(gradient(model_batch, data_batch, theta)))
                    theta -= eta * gradient(model_batch, data_batch, theta)
            new_error = errorfunc(self.X, self.y, theta)
            #print("Error =", new_error)
            if new_error < sgd_error:
                sgd_error = new_error
                theta_best = theta

        #print(new_error)
        return theta_best

class logistic_reg:
    def __init__(self, lmb=0.1):
        #Theta is a working beta, used in sgd. self.beta is the beta found by the sgd.
        #self.classes = classes #The actual classifications in the training dataset, n.c matrix
        #self.predictors = predictors #The predictor set corresponding to those classifications, n.p matrix.
        #beta has dimensions p.c
        self.lmb = lmb
        #self.n = np.size(predictors, axis=0) #number of different objects to identify
        #self.p = np.size(predictors, axis=1) #number of predictors for each object
        #self.c = np.size(classes, axis=1) #number of classifications

    def expo(self, X, theta):
        print("xbet", np.max(X @ theta))
        print("exp", np.max(np.exp(X @ theta)))
        quit()
        return np.exp(X @ theta)

    def probability(self, X, theta):
        #print(X.shape)
        #print(theta.shape)
        prob = np.zeros((X.shape[0], theta.shape[1]+1))
        prob[:,:-1] = np.exp(X @ theta)/(1 + np.sum(np.exp(X @ theta), keepdims=True, axis=1))
        prob[:,-1] = 1/(1 + np.sum(np.exp(X @ theta), keepdims=False, axis=1))
        return prob

    def log_cost(self, X, y, theta):
        cost = - y * np.log(self.probability(X, theta))
        n = y.shape[0]
        return np.sum(cost)/n

    def log_cost_gradient(self, X, y, theta):
        return -X.T @ (y - self.probability(X, theta))[:,:-1]

    def log_cost_L(self, X, y, theta):
        cost = - y * np.log(self.probability(X, theta)) + self.lmb*np.sum(theta**2)
        n = y.shape[0]
        return np.sum(cost)/n

    def log_cost_L_gradient(self, X, y, theta):
        return -X.T @ (y - self.probability(X, theta))[:,:-1] + 2*self.lmb*theta
    """
    def double_gradient_logreg(self, theta):
        return
    """
    def log_cost_num(self, X, y, theta):
        return - np.log(np.prod(np.sum(y * self.probability(X, theta), axis=1)) + pow(10,-10)) #"numerically determined" negative log cost, just multiplying the probabilities with the true classifications to remove the erroneous ones, summing to make a vector, multiplying all the vectors together to get the probability of all guesses being right, then negative log.

    def log_solver(self, X, y, minibatch_size, n_epochs = 50, max_iter = 50, t0 = 0.1, t1 = 50):
        gradient = gradient_descent(X, y, minibatch_size, n_epochs, max_iter, t0, t1)
        self.beta = gradient.sgd(self.log_cost_L, self.log_cost_L_gradient)
        return self.beta

    def log_prediction(self, X):
        return self.probability(X, self.beta)

    def log_confusion(self, prediction, y, labels, names=["Value", "Value"], title="Confusion Matrix"):
        n_data = np.size(y, axis=0)
        n_classes = np.size(y, axis=1)
        #prediction = self.probability(X, self.beta)
        confusion = np.zeros((n_classes,n_classes))
        true_answers = np.zeros((n_classes,n_classes))

        for i in range(n_data):
            k = np.argmax(y[i])
            l = np.argmax(prediction[i])
            confusion[k,l] += 1
            true_answers[k] += 1

        confusion /= true_answers
        print(confusion)
        confusion_matrix = pd.DataFrame(confusion, columns=labels)
        plt.figure(figsize=(15,8))
        #sns.set(font_scale=1.0)
        ax = sns.heatmap(data=confusion_matrix, annot=True, cbar_kws={'label': ''}, cmap="Blues", annot_kws={'size':14})
        ax.set_xlabel("Predicted "+names[0], fontsize=14)
        ax.set_ylabel("True "+names[1], fontsize=14)
        #(ylabel="True "+names[0], xlabel="Predicted "+names[1], fontsize=12)
        plt.title(title + f"    n={n_data}", fontsize=18)
        plt.show()
        return

class regression:
    def __init__(self, x, y, data, noise, n):
        self.x = x
        self.y = y
        self.data = data + noise
        self.noise = noise
        self.n = n
        self.lmb = 0

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
            plt.title("Bias-Variance Tradeoff Ridge with $\lambda$ = " + str(lmb))
            fig.savefig("Bias-Variance Tradeoff Ridge_n_" + str(self.n) + "_lmb_" + str(lmb))
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
        lmb_count = 100
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
        #ax.set_yscale('log')
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

    def set_lmb(self, lmb, printout=False):
        self.lmb = lmb
        if printout:
            print(f"Penalty parameter set to {self.lmb}")
        return

    def MSE_(self, X, y, beta):
        N = np.size(y)
        return (1.0 / N) * np.sum((X @ beta - y)**2) + self.lmb*np.sum(beta**2)

    def MSE_gradient(self, X, y, beta):
        N = np.size(y)
        return (2.0 / N) * (X.T @ (X @ beta - y)) + 2*self.lmb*np.abs(beta)

    def R2_(self, X, y, beta):
        return 1.0 - (np.sum((y - X @ beta)**2) / np.sum((y - np.mean(y))**2))

    def lin_solver(self, X, y, minibatch_size, n_epochs = 50, max_iter = 50, t0 = 0.1, t1 = 50):
        gradient = gradient_descent(X, y, minibatch_size, n_epochs, max_iter, t0, t1)
        self.beta = gradient.sgd(self.MSE_, self.MSE_gradient)
        return self.beta

    def scale_data_(self, X):
        # Scale two matricses whose first column both contain only ones
        # X1 is training data, X2 is testing data
        #np.newaxis = None

        if (len(X[0]) == 1):
            return X
        else:
            X = np.delete(X, 0, axis=1)

            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)

            X = np.concatenate((np.ones(len(X))[:, None], X), axis=1)
            return X


    def plot_MSE_sgd_minibatch(self, X_train, y_train, X_test, y_test):
        minibatch_range_count = int(np.log2(self.n))+1
        minibatch_sizes = np.logspace(0, minibatch_range_count-1, num=minibatch_range_count, base=2.0, dtype=int)

        print(minibatch_sizes)

        mse_minibatch = np.zeros(minibatch_range_count)
        r2_minibatch = np.zeros(minibatch_range_count)
        n_minibatches = np.floor(self.n/minibatch_sizes).astype(int)

        print(n_minibatches)

        t_minibatch = np.zeros(minibatch_range_count)

        for (minibatch_size, i) in zip(minibatch_sizes, range(minibatch_range_count)):
            t_0 = time()
            self.lin_solver(X_train, y_train, minibatch_size, n_epochs = 50, max_iter = 50, t0 = 0.1, t1 = 50)
            mse_minibatch[i] = self.MSE_(X_test, y_test, self.beta)
            r2_minibatch[i] = self.R2_(X_test, y_test, self.beta)
            t_minibatch[i] = time() - t_0
            print(f"Runtime: {t_minibatch[i]} sec w/ {n_minibatches[i]} minibatches")

            print(f"MSE: {mse_minibatch[i]}")
            print(f"R2 Score: {r2_minibatch[i]}")

        #compare with OLS analytic values
        beta_ols = self.ols_beta(X_train, y_train)
        mse_ols = self.MSE_(X_test, y_test, beta_ols)
        r2_ols = self.R2_(X_test, y_test, beta_ols)

        fig, ax1 = plt.subplots()

        color = "tab:blue"
        ax1.set_xlabel("Minibatch Count")
        ax1.set_ylabel("MSE", color=color)
        ax1.plot(n_minibatches, mse_minibatch, color=color, label="MSE SGD")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, np.max(mse_minibatch)*1.1)
        ax1.set_xscale('log', base=2)
        ax1.axhline(y=mse_ols, color=color, linestyle='--', label="MSE analytic")
        ax1.legend(loc=0)

        ax2 = ax1.twinx()

        color = "tab:red"
        """
        ax2.set_ylabel("Time [s]", color=color)
        ax2.plot(n_minibatches, t_minibatch, color=color, label="Runtime")
        ax2.tick_params(axis='y', labelcolor=color)
        #plt.xlim(0, max_degree - 1)
        plt.title("MSE and time spent for different minibatch counts")
        """
        ax2.set_ylabel("R2 Score", color=color)
        ax2.plot(n_minibatches, r2_minibatch, color=color, label="R2 Score SGD")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.1, 1.1)
        ax2.axhline(y=r2_ols, color=color, linestyle='--', label="R2 Score analytic")

        plt.title("MSE & R2 Score for minibatch counts")

        ax2.legend(loc=0)
        plt.show()

        return

    def plot_MSE_sgd_eta(self, X_train, y_train, X_test, y_test):

        n_t0 = 100
        n_t1 = 3
        eta0 = np.linspace(0.01, 1, n_t0)
        eta1 = np.array((10,50,10000))
        eta1[-1] = pow(10,5)

        mse_eta = np.zeros((n_t1,n_t0))
        r2_eta = np.zeros((n_t1,n_t0))
        t_eta = np.zeros((n_t1,n_t0))

        for i in range(n_t1):
            for j in range(n_t0):
                t_0 = time()
                self.lin_solver(X_train, y_train, minibatch_size=int(np.floor(np.size(y_train, axis=0)/4)), n_epochs = 50, max_iter = 50, t0 = eta0[j], t1 = eta1[i])
                mse_eta[i,j] = self.MSE_(X_test, y_test, self.beta)
                r2_eta[i,j] = self.R2_(X_test, y_test, self.beta)
                t_eta[i,j] = time() - t_0
                print(f"Runtime: {t_eta[i,j]:.2f} sec w/ t0 = {eta0[j]:.2f}, t1 = {eta1[i]:.2f}")

                print(f"MSE: {mse_eta[i,j]}")
                print(f"R2 Score: {r2_eta[i,j]}")

        #compare with OLS analytic values
        beta_ols = self.ols_beta(X_train, y_train)
        mse_ols = self.MSE_(X_test, y_test, beta_ols)
        r2_ols = self.R2_(X_test, y_test, beta_ols)

        fig, ax1 = plt.subplots()

        color = "tab:blue"
        ax1.set_xlabel("Learning Rate t0")
        ax1.set_ylabel("MSE", color=color)
        for i in range(n_t1):
            ax1.plot(eta0, mse_eta[i,:], label=f"MSE SGD, t1 = {eta1[i]}")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.1)
        ax1.axhline(y=mse_ols, color=color, linestyle='--', label="MSE analytic")
        ax1.legend(loc=0)
        """
        ax2 = ax1.twinx()

        color = "tab:red"

        ax2.set_ylabel("R2 Score", color=color)
        for i in range(n_t1):
            ax2.plot(eta0, r2_eta[i,:], color=color, label=f"R2 SGD, t1 = {eta1[i]}")
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(-0.1, 1.1)
        ax2.axhline(y=r2_ols, color=color, linestyle='--', label="R2 Score analytic")
        ax2.legend(loc=0)
        """
        plt.title("MSE - Learning Rate")


        plt.show()

        return

    def k_fold_sgd(self, x, splits = 5, shuffle = False):

        indices = np.arange(x.shape[0])
        if shuffle == True:
            rng = np.random.default_rng()
            rng.shuffle(indices)

        test_inds = np.array_split(indices, splits)
        train_inds = np.array_split(indices, splits)
        for i in range(splits):
            train_inds[i] = np.concatenate(np.delete(test_inds, i, 0))

        return test_inds, train_inds

    def cross_validation_sgd(self, X, y, splits):

        minibatch_size = int(self.n/8.0)
        test_inds, train_inds = self.k_fold(X, splits)
        lmb_count = 15
        lmb = np.logspace(-6, -1, lmb_count)
        MSE_kfold_ridge = np.zeros((lmb_count,splits))
        #MSE_kfold_lasso = np.zeros((lmb_count,splits))
        MSE_kfold_ols = np.zeros((lmb_count,splits))

        R2_kfold_ridge = np.zeros((lmb_count,splits))
        #R2_kfold_lasso = np.zeros((lmb_count,splits))
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
                    self.set_lmb(0)
                    beta_kfold_ols = self.lin_solver(X_train_kfold, y_train_kfold, minibatch_size=minibatch_size)
                    MSE_kfold_ols[i,j] = self.MSE_(X_test_kfold, y_test_kfold, beta_kfold_ols)
                    R2_kfold_ols[i,j] = self.R2_(X_test_kfold, y_test_kfold, beta_kfold_ols)

                self.set_lmb(lmb[i])

                beta_kfold_ridge = self.lin_solver(X_train_kfold, y_train_kfold, minibatch_size=minibatch_size)
                MSE_kfold_ridge[i,j] = self.MSE_(X_test_kfold, y_test_kfold, beta_kfold_ridge)
                R2_kfold_ridge[i,j] = self.R2_(X_test_kfold, y_test_kfold, beta_kfold_ridge)

                """
                _, _, y_pred_kfold_lasso = self.lasso(X_train_kfold, X_test_kfold, y_train_kfold, lmb[i])
                MSE_kfold_lasso[i,j] = self.MSE(y_test_kfold, y_pred_kfold_lasso)
                R2_kfold_lasso[i,j] = self.R2(y_test_kfold,y_pred_kfold_lasso)
                """

        MSE_kfold_ols = np.mean(MSE_kfold_ols, axis=1)
        MSE_kfold_ols[:] = MSE_kfold_ols[0]
        MSE_kfold_ridge = np.mean(MSE_kfold_ridge, axis=1)
        #MSE_kfold_lasso = np.mean(MSE_kfold_lasso, axis=1)

        R2_kfold_ols = np.mean(R2_kfold_ols, axis=1)
        R2_kfold_ols[:] = R2_kfold_ols[0]
        R2_kfold_ridge = np.mean(R2_kfold_ridge, axis=1)
        #R2_kfold_lasso = np.mean(R2_kfold_lasso, axis=1)


        fig, ax = plt.subplots()
        ax.plot(lmb, MSE_kfold_ols, label = "Ordinary Least Squares")
        ax.plot(lmb, MSE_kfold_ridge, label = "Ridge Regression")
        #ax.plot(lmb, MSE_kfold_lasso, label = "Lasso Regression")

        plt.legend()
        #ax.set_yscale('log')
        ax.set_xscale('log')

        plt.xlabel("Hyperparameter $\lambda$")
        plt.ylabel("Estimated MSE")
        plt.title("MSE with k-fold cross validation")
        plt.xlim(lmb[0], lmb[-1]+1)
        plt.ylim(0.8*np.min(np.array((np.min(MSE_kfold_ridge), np.min(MSE_kfold_ols)))), 1.2*np.max(np.array((np.max(MSE_kfold_ridge), np.max(MSE_kfold_ols)))))
        plt.show()
        #fig.savefig("K-fold-MSE_n_" + str(self.n))

        fig2, ax2 = plt.subplots()
        ax2.plot(lmb, R2_kfold_ols, label = "Ordinary Least Squares")
        ax2.plot(lmb, R2_kfold_ridge, label = "Ridge Regression")
        #ax2.plot(lmb, R2_kfold_lasso, label = "Lasso Regression")

        plt.legend()
        ax2.set_xscale('log')

        plt.xlabel("Hyperparameter $\lambda$")
        plt.ylabel("Estimated R2 Score")
        plt.title("R2 Score with k-fold cross validation")
        plt.xlim(lmb[0], lmb[-1]+1)
        plt.ylim(-0.5, 1.1)
        plt.show()
        #fig2.savefig("K-fold-R2_n_" + str(self.n))

        return


    def bootstrapResample_sgd(self, x, y):
        inds = np.random.randint(0, x.shape[0], size = x.shape[0])
        x_boot = x[inds]
        y_boot = y[inds]
        return x_boot, y_boot

    def bootstrapBiasVariance_sgd(self, X_train, y_train, X_test, y_test, n_boot, minibatch_size):
        y_pred = np.zeros((y_test.shape[0], n_boot))
        for i in range(n_boot):
            # Resample the data n_boot times, making a new prediction for each resampling.
            X_resampled, y_resampled = self.bootstrapResample(X_train, y_train)
            beta_resampled = self.lin_solver(X_resampled, y_resampled, minibatch_size)
            #print(beta_resampled.shape)
            y_pred[:,i] = (X_test @ beta_resampled)[:,0]

        y_test = y_test.reshape(-1,1)
        error = np.mean( np.mean((y_test - y_pred)**2, axis=1, keepdims=True) )
        bias = np.mean( (y_test - np.mean(y_pred, axis=1, keepdims=True))**2 )
        variance = np.mean( np.var(y_pred, axis=1, keepdims=True))

        print("Error: ", error)
        print("Bias: ", bias)
        print("Variance: ", variance, "\n")

        return error, bias, variance
