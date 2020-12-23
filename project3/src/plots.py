import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl

mpl.style.use('ggplot')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 3, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2, 'figure.titlesize': fontsize + 5}
plt.rcParams.update(newparams)

def plot3D(X, T, NN, A, diff):
    pad = 20
    fig = plt.figure(figsize=plt.figaspect(1.5))

    ax = fig.add_subplot(2, 1, 1, projection="3d")
    ax.set_title('Neural network approximation')
    s = ax.plot_surface(X, T, NN, linewidth=0, antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r'$t$', labelpad=pad)
    ax.set_ylabel(r'$x$', labelpad=pad)
    ax.set_zlabel(r'$u(x,t)$', labelpad=pad)

    ax = fig.add_subplot(2, 1, 2, projection="3d")
    ax.set_title(r'Absolute error $\epsilon$')
    s = ax.plot_surface(X, T, diff, linewidth=0, antialiased=False, cmap=cm.viridis)
    ax.set_xlabel(r'$t$', labelpad=pad)
    ax.set_ylabel(r'$x$', labelpad=pad)
    ax.set_zlabel(r'$\epsilon$', labelpad=pad)

    plt.show()


def plot2D(x, t, res1, res2, res3):
    # Plot the slices
    plt.figure(figsize=(10, 10))
    plt.title("Computed solutions at t = %g" % t[0])
    plt.plot(x, res1[0])
    plt.plot(x, res1[1])
    plt.legend(['Neural network', 'Analytical'])

    plt.figure(figsize=(10, 10))
    plt.title("Computed solutions at t = %g" % t[1])
    plt.plot(x, res2[0])
    plt.plot(x, res2[1])
    plt.legend(['Neural network', 'Analytical'])

    plt.figure(figsize=(10, 10))
    plt.title("Computed solutions at t = %g" % t[2])
    plt.plot(x, res3[0])
    plt.plot(x, res3[1])
    plt.legend(['Neural network', 'Analytical'])

    plt.show()