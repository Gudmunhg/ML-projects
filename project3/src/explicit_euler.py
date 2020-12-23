import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from plots import *

mpl.style.use('ggplot')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 3, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': False,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2, 'figure.titlesize': fontsize + 5}
plt.rcParams.update(newparams)

def g(x):
    return np.sin(np.pi * x)

def forward_euler(x, t, alpha, times):
    u = np.zeros(len(x))   # solution array
    u_1 = np.zeros(len(x))
    u_1[:] = g(x[:])
    Nx = len(x) - 1

    stored_solution = np.zeros([len(times), len(x)])

    stored = 0
    for time in t:
        u[1:Nx] = u_1[1:Nx] + alpha * (u_1[0:Nx - 1] - 2 * u_1[1:Nx] + u_1[2:Nx + 1])

        if time in times:
            stored_solution[stored] = u
            stored += 1

        u[0] = 0
        u[Nx] = 0

        u_1, u = u, u_1

    return stored_solution


def analytic(x, t):
    return (np.exp(-(np.pi**2) * t)* np.sin(np.pi * x))



def produce_results(dx, dt, L, T):
    x = np.arange(0, L, dx)
    t = np.arange(0, T, dt)
    alpha = dt / dx**2 
    times = [t[2], t[-1]]

    stored_solution = forward_euler(x, t, alpha, times)

    print("Max diff between analytical solution and numeric approximation using forward Euler with dx = %.2f :" % dx)

    fig, axs = plt.subplots(len(times), sharex=True)
    fig.suptitle(r"Forward Euler: $\Delta x = %.3f$" % dx)

    for i in range(len(times)):
        a_s = analytic(x, times[i])

        axs[i].plot(x, stored_solution[i], label="Numerical", linewidth=5)
        axs[i].plot(x, a_s, label="Analytic", linewidth=5)
        axs[i].set_title(r"$t = %.2f$" % times[i])

        if a_s[int(len(x)/2)] < 0.1:
            axs[i].yaxis.set_major_formatter(FormatStrFormatter('%.e'))

        print("diff = ", np.max(np.abs(a_s - stored_solution[i])), "with t = {:2f}".format(times[i]))
        #print((1.0/(len(x)**2)) * (stored_solution[i] - a_s)**2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.legend()
    plt.xlabel(r'$x$')
    plt.show()

    store = forward_euler(x, t, alpha, t)
    X, T = np.meshgrid(x, t)
    a = analytic(X, T)

    plot3D(X, T, store, a, np.abs(store - a))


L = 1
T = 1

dx = 0.1
dt = 0.5 * dx**2
produce_results(dx, dt, L, T)

dx = 0.01
dt = 0.5 * dx**2
produce_results(dx, dt, L, T)