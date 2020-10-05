import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

# A seed just to ensure that the random numbers are the same for every run.
# Useful for eventual debugging.
seed = 3155#np.random.randint(0,99999999)
np.random.seed(seed)

# Generate the data.
nsamples = 100
x = np.random.randn(nsamples)
y = 3*x**2 + np.random.randn(nsamples)

## Cross-validation on Ridge regression using KFold only

# Decide degree on polynomial to fit
poly = PolynomialFeatures(degree = 6)

# Decide which values of lambda to use
nlambdas = 500
lambdas = np.logspace(-3, 5, nlambdas)

# Initialize a KFold instance

k = 5

kfold = KFold(n_splits = k, shuffle=False)
#print(kfold.split(x))

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((nlambdas, k))


def k_fold(x, splits = 5, shuffle = False):

    indices = np.arange(x.shape[0])
    if shuffle == True:
        rng = np.random.default_rng()
        rng.shuffle(indices)

    test_inds = np.array_split(indices, splits)
    train_inds = np.array_split(indices, splits)
    for i in range(splits):
        train_inds[i] = np.concatenate(np.delete(test_inds, i, 0))

    return test_inds, train_inds

i = 0
test_inds, train_inds = k_fold(x, splits = k, shuffle=False)
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)

    for j in range(k):
        xtrain = x[train_inds[j]]
        ytrain = y[train_inds[j]]

        xtest = x[test_inds[j]]
        ytest = y[test_inds[j]]

        Xtrain = poly.fit_transform(xtrain[:, np.newaxis])
        ridge.fit(Xtrain, ytrain[:, np.newaxis])

        Xtest = poly.fit_transform(xtest[:, np.newaxis])
        ypred = ridge.predict(Xtest)

        scores_KFold[i,j] = np.sum((ypred - ytest[:, np.newaxis])**2)/np.size(ypred)

        #j += 1
    i += 1


estimated_mse_KFold = np.mean(scores_KFold, axis = 1)

## Cross-validation using cross_val_score from sklearn along with KFold

# kfold is an instance initialized above as:
# kfold = KFold(n_splits = k)

estimated_mse_sklearn = np.zeros(nlambdas)
i = 0
for lmb in lambdas:
    ridge = Ridge(alpha = lmb)

    X = poly.fit_transform(x[:, np.newaxis])
    estimated_mse_folds = cross_val_score(ridge, X, y[:, np.newaxis], scoring='neg_mean_squared_error', cv=kfold)

    # cross_val_score return an array containing the estimated negative mse for every fold.
    # we have to the the mean of every array in order to get an estimate of the mse of the model
    estimated_mse_sklearn[i] = np.mean(-estimated_mse_folds)

    i += 1

## Plot and compare the slightly different ways to perform cross-validation


plt.figure()

plt.plot(np.log10(lambdas), estimated_mse_sklearn, label = 'cross_val_score')
plt.plot(np.log10(lambdas), estimated_mse_KFold, 'r--', label = 'KFold')

plt.xlabel('log10(lambda)')
plt.ylabel('mse')

plt.legend()

plt.show()
