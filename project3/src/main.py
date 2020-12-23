import network as nn
import numpy as np
import tensorflow.compat.v1 as tf
from plots import *
import os

mpl.style.use('ggplot')

fontsize = 20
newparams = {'axes.titlesize': fontsize + 3, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'upper left',
             'legend.fontsize': fontsize + 2, 'figure.titlesize': fontsize + 5}
plt.rcParams.update(newparams)

src_path = os.getcwd()
os.chdir(src_path[:-3])
path = os.getcwd() + "\\results\\"


def run_diffusion_with_diff_iters(num_iter):
    errors = np.zeros((len(num_iter), 2))

    for sol in range(len(num_iter)):
        s, a_s, mse = nn.solve_diffusion_with_network(10, 10, [300], opt=tf.train.AdamOptimizer, lr=0.001, num_iter=num_iter[sol])
        errors[sol][0] = num_iter[sol]
        errors[sol][1] = np.max(np.abs(a_s - s))

    np.savetxt(path + "var_iter\\mse_var_iters.txt", errors)


def run_diffusion_with_diff_optimizers():
    optimizers = [tf.train.AdadeltaOptimizer, tf.train.AdagradOptimizer, tf.train.AdamOptimizer, tf.train.GradientDescentOptimizer, tf.train.RMSPropOptimizer]
    opt_str = ["Adadelta", "Adagrad", "Adam", "GradientDescent", "RMSProp"]
    mses = np.zeros((100000, len(optimizers)))

    for o in range(len(optimizers)):
        s, a_s, mse = nn.solve_diffusion_with_network(10, 10, [300], opt=optimizers[o], lr=0.001, num_iter=100000)
        mses[:, o] = mse

    np.savetxt(path + "optimiser\\mse_optimisers.txt", mses)


def run_optimal_diffusion(Nx, Nt, num_iter):
    s, a, mse = nn.solve_diffusion_with_network(Nx, Nt, [300], opt=tf.train.AdamOptimizer, lr=0.001, num_iter=num_iter)

    s = s.reshape((Nt, Nx))
    a = a.reshape((Nt, Nx))

    np.savetxt(path + "optimal\\solution.txt", s)
    np.savetxt(path + "optimal\\analytical.txt", a)
    np.savetxt(path + "optimal\\mse_optimal.txt", mse)


def load_results(*results):
    pass


def load_var_optimisers(path):
    path = path + "\\optimiser\\"
    mses = np.loadtxt(path + "mse_optimisers.txt")
    opt_str = ["Adadelta", "Adagrad", "Adam", "GradientDescent", "RMSProp"]

    iters = range(0, len(mses[:, 0]))
    for i in range(mses.shape[1]):
        plt.semilogy(iters, mses[:, i], label=opt_str[i])

    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left")
    plt.title("Comparison of optimisers")
    plt.show()

def load_var_iters(path):
    path = path + "var_iter\\"
    errors = np.loadtxt(path + "mse_var_iters.txt")

    plt.semilogx(errors[:, 0], errors[:, 1], '-o')
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.title("MSE using ADAM w.r.t iterations")
    plt.show()

def load_optimal_results(path):
    path = path + "\\optimal\\"
    s = np.loadtxt(path + "solution.txt")
    a = np.loadtxt(path + "analytical.txt")
    mse = np.loadtxt(path + "mse_optimal.txt")

    x = np.linspace(0, 1, len(s[:, 0]))
    t = np.linspace(0, 1, len(s[0, :]))

    X, T = np.meshgrid(x, t)
    plot3D(X, T, s, a, np.abs(a - s))



def get_eigen_results():
    matrix_size = 6
    A, x_0, eigen_vals = nn.eigen_data(matrix_size)

    print("A = \n", A)

    print("Numpy eigenvalues: \n", eigen_vals)

    eigen_max = nn.eigenvalues_with_network(A, x_0, matrix_size)

    eigen_min = nn.eigenvalues_with_network(-A, x_0, matrix_size,
                                         num_iter=50000, layers=[10], act=tf.nn.tanh, min=True)

    abs_error_max = abs(np.max(eigen_vals) - eigen_max)
    abs_error_min = abs(np.min(eigen_vals) - eigen_min)

    print("The absolute error of NN model compared with numpy is {} for lambda_max and {} for lambda_min".format(
        abs_error_max, abs_error_min))

    print("The relative error is {} for lambda_max and {} for lambda_min".format(
        abs_error_max / np.max(eigen_vals), abs_error_min / np.min(eigen_vals)))



#run_diffusion_with_diff_iters([10, 100, 1000, 10000, 100000])
#run_diffusion_with_diff_optimizers()
#run_optimal_diffusion(20, 20, 100000)

#load_var_optimisers(path)
#load_var_iters(path)
#load_optimal_results(path)
get_eigen_results()
