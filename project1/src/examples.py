from regression import regression
import numpy as np


# Examples of how to use the regression class and its methods
np.random.seed(1111)
n = 1000
# noise uncertainty
sigma = 0.1

x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
#x, y = np.meshgrid(x, y)

noise = np.random.normal(0, sigma, n)
data = regression.FrankeFunction(x, y) + noise

p = 5

# Create regression class
# This class acts more like a container, at least for now,
# so all methods return values, and the class object
# does not remember many variables

# Create main class object
test = regression(x, y, data)
# Create feature matrix
X = test.create_feature_matrix(p)
# split data
X_train, X_test, y_train, y_test = test.split_data(X)
# Scale data
scaled_X_train, scaled_X_test = test.scale_data(X_train, X_test)


# OLS
# Create beta
beta = test.ols_beta(scaled_X_train, y_train)
# then train the data and get prediction
y_tilde, y_predict = test.make_prediction(scaled_X_train, scaled_X_test, beta)
# Write R2/MSE out to console
test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "OLS scores:")


# RIDGE
# Create ridge beta
ridge_beta = test.ridge_beta(scaled_X_train, y_train, lmb=1.9)

y_tilde, y_predict = test.make_prediction(
    scaled_X_train, scaled_X_test, ridge_beta)

test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "Ridge scores:")

# LASSO
m = 100
lmb = np.logspace(-4, 0, m)
lasso_beta, y_tilde, y_predict = test.lasso(
    scaled_X_train, scaled_X_test, y_train, lmb[67])
test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "Lasso scores:")