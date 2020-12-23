import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from typing import Callable
import matplotlib as mpl
from plots import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import tensordot as tdot
from tensorflow.compat.v1 import transpose as tp


tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

tf.set_random_seed(343)
np.random.seed(343)


"""
Neural Network implementation for solving (partial) differential equations.
"""

"""
Function for creating and preparing data for input into network.
Spesifically diffusion problems
"""
def diffusion_data(Nx, Nt):
    x_np = np.linspace(0, 1, Nx)
    t_np = np.linspace(0, 1, Nt)

    x_np, t_np = np.meshgrid(x_np, t_np)

    x = x_np.ravel()
    t = t_np.ravel()

    x = tf.reshape(tf.convert_to_tensor(x), shape=(-1, 1))
    t = tf.reshape(tf.convert_to_tensor(t), shape=(-1, 1))

    return x, t, tf.concat([x, t], 1)

"""
Function for creating and preparing data for input into network.
Spesifically eigenvalue problems
"""
def eigen_data(matrix_size=6):
    A = np.random.random_sample(size=(matrix_size, matrix_size))
    A = (A.T + A) / 2.0

    x_0 = np.random.random_sample(size=(1, matrix_size))

    eigen_vals, eigen_vecs = np.linalg.eig(A)

    return A, x_0, eigen_vals


"""
Prepare input and output layer, and also create hidden layers 
"""
def prepare_layers(num_hidden_neurons: list, input, activation=None, size=1):
    with tf.variable_scope('dnn'):
        nh = num_hidden_neurons
        nhl = np.size(nh)

        p_layer = input

        act = tf.nn.sigmoid
        if activation is not None:
            act = activation

        for layer in range(nhl):
            current_layer = tf.layers.dense(p_layer, nh[layer], activation=act)
            p_layer = current_layer

        dnn_output = tf.layers.dense(p_layer, size)

    return dnn_output

"""
Train the network for the spesific problem.
"""
def train(diffusion_params=None, eigen_params=None, num_iter=10000, lr=0.001, optimizer=None):
    with tf.name_scope('loss'):
        if (eigen_params is not None):
            out_l = eigen_params[0]
            A = eigen_params[1]
            matrix_size = eigen_params[2]

            x_trial = tf.transpose(out_l)
            I = np.eye(matrix_size)

            f1 = (tdot(tp(x_trial), x_trial, axes=1) * A)
            f2 = (1 - tdot(tp(x_trial), tdot(A, x_trial, axes=1), axes=1)) * I
            func = tdot((f1 - f2), x_trial, axes=1)

            func = tp(func)
            x_trial = tp(x_trial)

            loss = tf.losses.mean_squared_error(func, x_trial)
        else:
            out_l = diffusion_params[0]
            x = diffusion_params[1]
            t = diffusion_params[2]

            u_trial = (1 - t) * g(x) + x * (1 - x) * t * out_l

            u_trial_dt = tf.gradients(u_trial, t)
            u_trial_d2x = tf.gradients(tf.gradients(u_trial, x), x)

            zeros = tf.reshape(tf.convert_to_tensor(
                np.zeros(x.shape)), shape=(-1, 1))

            loss = tf.losses.mean_squared_error(
                zeros, u_trial_dt[0] - u_trial_d2x[0])

    mse = np.zeros(num_iter)
    with tf.name_scope('train'):
        opt = tf.train.AdamOptimizer(lr)
        #opt = tf.train.GradientDescentOptimizer(lr)

        if optimizer is not None:
            opt = optimizer(lr)

        operation = opt.minimize(loss)

    init = tf.global_variables_initializer()

    if eigen_params is not None:
        with tf.Session() as sess:
            init.run()
            for i in range(num_iter):
                sess.run(operation)

                mse[i] = loss.eval()

            x_dnn = x_trial.eval()
        return x_dnn

    else:
        with tf.Session() as sess:
            init.run()
            for i in range(num_iter):
                sess.run(operation)
                mse[i] = loss.eval()

            u_dnn = u_trial.eval()
            analytic_solution = (tf.exp(-(np.pi**2) * t)
                                 * tf.sin(np.pi * x)).eval()

        return u_dnn, analytic_solution, mse


def g(x):
    return tf.sin(np.pi * x)


def solve_diffusion_with_network(Nx, Nt, layers=[10], opt=tf.train.AdamOptimizer, lr=0.1, num_iter=10000):
    x, t, points = diffusion_data(Nx, Nt)
    dnn_output = prepare_layers(layers, points, activation=tf.nn.tanh)
    diffusion_params = [dnn_output, x, t]
    solution, analytic_solution, mse = train(diffusion_params=diffusion_params, optimizer=opt, lr=lr, num_iter=num_iter)

    tf.reset_default_graph()

    return solution, analytic_solution, mse


def eigenvalues_with_network(A, x_0, matrix_size, num_iter=10000, layers=[50], min=False, act=tf.nn.sigmoid, vector=False):
    start_matrix = A

    A = tf.convert_to_tensor(A)
    x_0 = tf.convert_to_tensor(x_0)

    output = prepare_layers(layers, x_0, size=matrix_size, activation=act)
    eigen_params = [output, A, matrix_size]
    x_dnn = train(eigen_params=eigen_params, num_iter=num_iter).T
    eigen_val_nn = (x_dnn.T @ start_matrix @ x_dnn) / (x_dnn.T @ x_dnn)

    if min is True:
        eigen_val_nn = -eigen_val_nn
        print("NN eigenvalue min: \n", eigen_val_nn)

    else:
        print("NN eigenvalue max: \n", eigen_val_nn)

    tf.reset_default_graph()

    if vector is True:
        return eigen_val_nn, x_dnn
    else:
        return eigen_val_nn

