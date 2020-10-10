from regression import regression
from plotter import error_plot, make_2plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.style.use('fivethirtyeight')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 5, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

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

p = 12

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


conf_analytic = test.analytic_confindence_interval(scaled_X_train, beta, noise)
np.set_printoptions(precision=6, suppress=True)
print("     lower       beta        upper")
print(conf_analytic)

# Plot beta with its respective upper and lower limits
error_plot(beta, conf_analytic[:, 0], conf_analytic[:, 2])

print("-------------------")
test.bootstrapBiasVariance(scaled_X_train, y_train, scaled_X_test, y_test, 100)

lmb, MSE_kfold_ols, MSE_kfold_ridge, MSE_kfold_lasso = test.cross_validation(
    X, splits=5)

fig, ax = plt.subplots()
ax.plot(lmb, MSE_kfold_ols, label="Ordinary Least Squares")
ax.plot(lmb, MSE_kfold_ridge, label="Ridge Regression")
ax.plot(lmb, MSE_kfold_lasso, label="Lasso Regression")

plt.legend()
# ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel(r"Hyperparameter $\lambda$")
plt.ylabel("Estimated MSE")
plt.title("MSE k-fold cross validation")
plt.xlim(lmb[0], lmb[-1] + 1)
plt.show()
fig.savefig("K-fold-MSE")"""