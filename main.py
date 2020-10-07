from regression import regression
from plotter import error_plot
import numpy as np

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4



n = 10000
np.random.seed(1111)
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
#split data
X_train, X_test, y_train, y_test = test.split_data(X)
#Scale data
scaled_X_train, scaled_X_test = test.scale_data(X_train, X_test)


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
lasso_beta, y_tilde, y_predict = test.lasso(scaled_X_train, scaled_X_test, y_train, lmb[67])
test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "Lasso scores:")

#Create a plot if the mean squared error of polynomials up to p + 1
test.make_MSE_plot(10 + 1)

print("-------------------")

conf_analytic = test.analytic_confindence_interval(scaled_X_train, beta)
np.set_printoptions(precision=6, suppress=True)
print("     lower       beta        upper")
print(conf_analytic)

#Plot beta with its respective upper and lower limits
error_plot(beta, conf_analytic[:, 0], conf_analytic[:, 2])