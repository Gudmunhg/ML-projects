from regression import regression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('fivethirtyeight')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 5, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15,10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best', 'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2)**2) - 0.25 * ((9 * y - 2)**2))
    term2 = 0.75 * np.exp(-((9 * x + 1)**2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7)**2 / 4.0 - 0.25 * ((9 * y - 3)**2))
    term4 = -0.2 * np.exp(-(9 * x - 4)**2 - (9 * y - 7)**2)
    return term1 + term2 + term3 + term4



n = 400
np.random.seed(1111)
# noise uncertainty
sigma = 0

#x = np.random.uniform(0, 1, n)
x = np.linspace(0, 1, n)
y = np.random.uniform(0, 1, n)

noise = np.random.normal(0, sigma, n)
data = FrankeFunction(x, y) + noise

#Create regression class
#This class acts more like a container, at least for now,
#so all methods return values, and the class object
#does not remember many variables

#Create main class object
test = regression(x, y, data, noise, n)
p_start = 0
p_stop = 10
p_range = np.arange(p_start,p_stop + 1)
ols_error = np.zeros(p_range.shape)
ols_bias = np.zeros(p_range.shape)
ols_variance = np.zeros(p_range.shape)

for p in p_range:
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
    #test.accuracy_printer(y_train, y_tilde, y_test, y_predict, "OLS scores:")

    print("p = ", p)

    ols_error[p-p_start], ols_bias[p-p_start], ols_variance[p-p_start] = test.bootstrapBiasVariance(scaled_X_train, y_train, scaled_X_test, y_test, 10000, lmb = 0)

fig, ax = plt.subplots()

ax.plot(p_range, ols_error, label = 'ols error')
ax.plot(p_range, ols_bias, label = 'ols bias')
ax.plot(p_range, ols_variance, label = 'ols variance')
ax.set_yscale('log')
plt.legend()
plt.show()
