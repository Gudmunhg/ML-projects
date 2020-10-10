from regression import regression
from plotter import error_plot, make_2plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from imageio import imread

mpl.style.use('fivethirtyeight')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 3, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2, 'figure.titlesize': fontsize + 5}
plt.rcParams.update(newparams)


def plot3D(row, col, data, title, subplot=False):
    fig = plt.figure()
    if (subplot):
        ax = fig.subplots(1, len(data), 1, projection='3d')
    else:
        ax = fig.gca(projection='3d')

    for i in range(len(data)):
        surf = ax.plot_surface(col, row, data, cmap=cm.viridis, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


#File for producing results/figures
#See example.py for examples of class method usage
"""
What results/figures do we want?
Hastie figure MSE
"""
def normalise(v):
    return (v - np.min(v))/(np.max(v) - np.min(v))

def prepare_data(X, beta, row, col):
    data = regression.make_single_prediction(X, beta)
    data = data.reshape(row, col)
    data = normalise(data)
    return data

def get_mesh(row, col):
    ax_row = np.random.uniform(0, 1, size=row)
    ax_col = np.random.uniform(0, 1, size=col)

    ax_row_sorted = np.sort(ax_row)
    ax_col_sorted = np.sort(ax_col)

    colmat, rowmat = np.meshgrid(ax_col_sorted, ax_row_sorted)
    return rowmat, colmat

def plot_terrain(p):
    #N = 100
    terrain1 = imread('DataFiles/SRTM_data_Norway_1.tif')
    #terrain1 = terrain1[:N,:N]
    row = terrain1.shape[0]
    col = terrain1.shape[1]
    rowmat, colmat = get_mesh(row, col)

    data = terrain1
    data = normalise(data)

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')

    surf = ax.plot_surface(colmat, rowmat, data, cmap=cm.terrain, linewidth=0, antialiased=False)
    plt.title("Test data", y = -0.17)
    row_arr = rowmat.ravel()
    col_arr = colmat.ravel()
    data_arr = data.ravel()

    reg = regression(row_arr, col_arr, data)

    X = reg.create_feature_matrix(p)

    X_train, X_test, y_train, y_test = reg.split_data(X, ratio=0.2)
    scaled_X_train, scaled_X_test = reg.scale_data(X_train, X_test)

    ols_beta = reg.ols_beta(scaled_X_train, y_train)
    ridge_beta = reg.ridge_beta(scaled_X_train, y_train, 1e-3)
    lasso_beta, _, _ = reg.lasso(scaled_X_train, scaled_X_test, y_train, 1e-4)

    data_pred_ols = prepare_data(X, ols_beta, row, col)

    data_pred_ridge = prepare_data(X, ridge_beta, row, col)

    data_pred_lasso = lasso_beta.predict(X)
    data_pred_lasso = normalise(data_pred_lasso.reshape(row, col))

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(colmat, rowmat, data_pred_ols, cmap=cm.terrain, linewidth=0, antialiased=False)
    plt.title("OLS", y = -0.17)

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax.plot_surface(colmat, rowmat, data_pred_ridge, cmap=cm.terrain, linewidth=0, antialiased=False)
    plt.title("Ridge", y = -0.17)

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    surf = ax.plot_surface(colmat, rowmat, data_pred_lasso, cmap=cm.terrain, linewidth=0, antialiased=False)
    plt.title("Lasso", y = -0.17)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.suptitle("Terrain test data vs predicition")
    plt.show()

#Warning: Running this will take some time
#Consider uncommenting code to run with less values
#plot_terrain(5)

def plot_franke(row, col):
    noise = np.random.normal(0, 0.05, (row, col))

    rowmat, colmat = get_mesh(row, col)
    franke = regression.FrankeFunction(colmat, rowmat) + noise

    fig = plt.figure()
    ax = fig.add_subplot(2,2,1, projection='3d')
    surf = ax.plot_surface(colmat, rowmat, franke, cmap=cm.viridis, linewidth=0, antialiased=False)

    reg = regression(rowmat, colmat, franke)

    num = 2
    for i in [5, 11, 12]:
        X = reg.create_feature_matrix(i)
        X_train, X_test, y_train, y_test = reg.split_data(X, ratio=0.2)
        scaled_X_train, scaled_X_test = reg.scale_data(X_train, X_test)
        ols_beta = reg.ols_beta(scaled_X_train, y_train)

        data_plot = prepare_data(X, ols_beta, row, col)
        ax = fig.add_subplot(2, 2, num, projection='3d')
        surf = ax.plot_surface(colmat, rowmat, data_plot, cmap=cm.viridis, linewidth=0, antialiased=False)
        plt.title(r"$p = %d$" % i, y = -0.17)
        num += 1

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.suptitle("Franke function vs predictions using OLS")
    plt.show()

#plot_franke(100, 200)

np.random.seed(1111)
n = 1000
# noise uncertainty
sigma = 0.1

x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
#x, y = np.meshgrid(x, y)

noise = np.random.normal(0, sigma, n)
data = regression.FrankeFunction(x, y) + noise

reg = regression(x, y, data)

poly_max = 12
poly, train_error, test_error = reg.make_MSE_comparison(poly_max)
title = "Mean squared error of training vs testing data"
xlabel = "Model Complexity"
ylabel = "Prediction Error"
make_2plot(poly, train_error, test_error, poly_max,
           "Train Error", "Test Error", xlabel, ylabel, title)