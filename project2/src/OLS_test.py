from log_lin_sgd import regression
import numpy as np
from matplotlib import pyplot as plt

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4

n = 4096
#np.random.seed(2020)
# noise uncertainty
sigma = 0.1

x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)

noise = np.random.normal(0, sigma, n)
data = FrankeFunction(x, y) + noise
p = 5


#Create regression class
#This class acts more like a container, at least for now,
#so all methods return values, and the class object
#does not remember many variables

#Create main class object
test = regression(x, y, data, noise, n)
#Create feature matrix
X = test.create_feature_matrix(p)
#scale data
X_scaled = test.scale_data_(X)
#split data
X_train, X_test, y_train, y_test = test.split_data(X_scaled)
#Scale data
#scaled_X_train, scaled_X_test = test.scale_data_(X_train), test.scale_data_(X_test)

y_train, y_test = y_train.reshape((-1,1)), y_test.reshape((-1,1))

test.set_lmb(0.000, printout=True)
#test.plot_MSE_sgd_minibatch(X_train, y_train, X_test, y_test)
#test.plot_MSE_sgd_eta(X_train, y_train, X_test, y_test)

test.cross_validation_sgd(X_scaled, data.reshape((-1,1)), splits=5)
#minibatch_size = 50
#n_boot = 10
#test.bootstrapBiasVariance_sgd(scaled_X_train, y_train.reshape((-1,1)), scaled_X_test, y_test.reshape((-1,1)), n_boot, minibatch_size)

"""
n_iter = 50
#t = np.logspace(0, 3, num=n_iter, dtype=int)
t = np.linspace(10,n, n_iter, dtype=int)
print(t)
error_sgd = np.zeros(n_iter)
error_sgd_best = pow(10,10)

error_ols = np.zeros(n_iter)

for i in range(n_iter):
    franke_sgd = gradient_descent(scaled_X_train, y_train.reshape((-1, 1)), minibatch_size=10, n_epochs = t[i], max_iter = 50, t0 = int(t[i]*n/10), t1 = int(t[i]*n*30/10))
    beta_sgd = franke_sgd.sgd(MSE, gradient_lin_reg).reshape((-1))
    error_sgd[i] = MSE(scaled_X_test, y_test, beta_sgd)
    if error_sgd[i] < error_sgd_best:
        error_sgd_best = error_sgd[i]
        beta_sgd_best = beta_sgd
    print(f"MSE SGD: {error_sgd[i]:.5f}")

beta_ols = test.ols_beta(scaled_X_train, y_train)
error_ols[:] = MSE(scaled_X_test, y_test, beta_ols)
print(f"MSE OLS inversion: {error_ols[0]:.5f}")
model_sgd = scaled_X_test @ beta_sgd_best
model_ols = scaled_X_test @ beta_ols

print(f"R2 SGD: {test.R2(model_sgd, y_test)} \nR2 OLS inversion: {test.R2(model_ols, y_test)}")
print(beta_sgd_best, beta_ols)

fig, ax = plt.subplots()
ax.plot(t, error_sgd, label="SGD")
ax.plot(t, error_ols, label="OLS inversion")
plt.legend()
plt.xlabel("Minibatch Size")
plt.ylabel("MSE")
plt.title(f"MSE OLS inversion and SGD with different minibatch size")
plt.show()
"""

"""
##OLS
#Create beta
beta = test.ols_beta(scaled_X_train, y_train)
#then train the data and get prediction
y_tilde, y_predict = test.make_prediction(scaled_X_train, scaled_X_test, beta)
#Write R2/errors out to console
test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "OLS scores:")


##RIDGE
#Create ridge beta
ridge_beta = test.ridge_beta(scaled_X_train, y_train, lmb=1.9)
#then train the data and get prediction
y_tilde, y_predict = test.make_prediction(scaled_X_train, scaled_X_test, ridge_beta)
#Write R2/errors out to console
test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "Ridge scores:")

##LASSO
m = 100
lmb = np.logspace(-4, 0, m)
lasso_beta, y_tilde, y_predict = test.lasso(scaled_X_train, scaled_X_test, y_train, lmb[45])
test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "Lasso scores:")

test.make_MSE_plot(10 + 1)

conf = test.confidence_interval(X, 20)
print("     lower       mean        upper")
print(conf)
print("---------")

t = test.bootstrap()
"""
#test.bootstrapBiasVariance(scaled_X_train, y_train, scaled_X_test, y_test, 1000)

#test.ridge_cross_validation(scaled_X_train, data, splits = 5)

#test.bias_variance_plot(0, 11, lmb = 0)
