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
newparams = {'axes.titlesize': fontsize + 5, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

#File for producing results/figures
#See example.py for examples
"""
What results/figures do we want?
"""
"""

n = 100
np.random.seed(1111)
# noise uncertainty
sigma = 0.1

x = np.random.uniform(0, 1, n)
y = np.random.uniform(0, 1, n)
x, y = np.meshgrid(x, y)

noise = np.random.normal(0, sigma, n)
data = regression.FrankeFunction(x, y) + noise
#print(data.shape)
#print(data)
#print(data.flatten())
#print(data.flatten().reshape((10,10)))"""
p = 10
terrain1 = imread('DataFiles/SRTM_data_Norway_2.tif')
# Generate the data
#nrow = 100
#ncol = 200
nrow = terrain1.shape[0]
ncol = terrain1.shape[1]

ax_row = np.random.uniform(0, 1, size=nrow)
ax_col = np.random.uniform(0, 1, size=ncol)

ind_sort_row = np.argsort(ax_row)
ind_sort_col = np.argsort(ax_col)

ax_row_sorted = np.sort(ax_row)
ax_col_sorted = np.sort(ax_col)

colmat, rowmat = np.meshgrid(ax_col_sorted, ax_row_sorted)

noise = np.random.normal(0, 0.1, (nrow, ncol))

#data = regression.FrankeFunction(rowmat, colmat) + noise
data = terrain1

row_arr = rowmat.ravel()
col_arr = colmat.ravel()
data_arr = data.ravel()

reg = regression(row_arr, col_arr, data)

X = reg.create_feature_matrix(p)
"""
X_train, X_test, y_train, y_test = reg.split_data(X, ratio=0.2)
scaled_X_train, scaled_X_test = reg.scale_data(X_train, X_test)
ols_beta_split = reg.ols_beta(scaled_X_train, y_train)
data_pred_split = reg.make_single_prediction(scaled_X_train, ols_beta_split)


ratio = data_pred_split.shape[0]/data_arr.shape[0]
print(ratio)
print(int(5/2), int(5/2) + 0.5)
#nrow_new = nrow * int((1 - 0.2)*)
nrow_new = 100
ncol_new = 160#int(ncol*0.5)
data_plot_split = data_pred_split.reshape(nrow_new, ncol_new)
print(data_plot_split.shape)


ax_row = np.random.uniform(0, 1, size=nrow_new)
ax_col = np.random.uniform(0, 1, size=ncol_new)

ax_row_sorted = np.sort(ax_row)
ax_col_sorted = np.sort(ax_col)

colmat_new, rowmat_new = np.meshgrid(ax_col_sorted, ax_row_sorted)
"""

ols_beta = reg.ols_beta(X, data_arr)
data_pred = reg.make_single_prediction(X, ols_beta)
data_plot = data_pred.reshape(nrow, ncol)
print(data_plot.shape)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(colmat, rowmat, data_plot, cmap=cm.viridis, linewidth=0, antialiased=False)
#surf = ax.plot_surface(colmat_new, rowmat_new, data_plot_split, cmap=cm.viridis, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Fitted Franke with p = 10')
plt.show()


"""
# Create main class object
reg = regression(x, y, data)
X = reg.create_feature_matrix(p)
ratio = 0.25
X_train, X_test, y_train, y_test = reg.split_data(X, ratio)
scaled_X_train, scaled_X_test = reg.scale_data(X_train, X_test)
ols_beta = reg.ols_beta(scaled_X_train, y_train)
ridge_beta = reg.ridge_beta(scaled_X_train, y_train, 1e-1)
y_tilde, y_pred = reg.make_prediction(scaled_X_train, scaled_X_test, ols_beta)
y_tilde, y_pred = reg.make_prediction(scaled_X_train, scaled_X_test, ridge_beta)


r = int(np.sqrt(ratio*n**2))

new_n = int(n*ratio)
x_num = np.linspace(0, 1, r)
y_num = np.linspace(0, 1, r)
x_num, y_num = np.meshgrid(x_num, y_num)
print(x_num.shape)
print(y_num.shape)
z = np.reshape(y_pred, (-1, r))
print(z.shape)

x = np.linspace(0, 1, 1000)
y = np.linspace(0, 1, 1000)
x, y = np.meshgrid(x, y)
print(x.shape)
print(y.shape)
franke = regression.FrankeFunction(x, y)
print(franke.shape)
#Plot of franke function in 3d
fig = plt.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, franke, cmap=cm.coolwarm)
surf = ax.plot_surface(x_num, y_num, z, cmap=cm.coolwarm)
plt.title('FrankeFunction')
plt.xlabel('x')
plt.ylabel('y')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


"""

