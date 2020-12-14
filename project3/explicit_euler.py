import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

def g(x):
    return np.sin(np.pi * x)


dx = 0.01
dt = 0.5 * dx**2
x = np.arange(0, 1, dx)
t = np.arange(0, 1, dt)

alpha = dt / dx**2

V = np.zeros(len(x))

u = np.zeros(len(x))
unew = np.zeros(len(x))

for i in range(1, len(x) - 1):
	u[i] = g(x[i])

for _ in range(1, len(t) - 1):
	for i in range(1, len(x) - 1):
		unew[i] = alpha * u[i - 1] + (1 - 2*alpha) * u[i] + alpha * u[i + 1]

analytic_solution = (np.exp(-(np.pi**2) * t)* np.sin(np.pi * x))
print(analytic_solution)

plt.plot(x, unew)
plt.show()

A = diags([alpha, 1 - 2*alpha, alpha], [-1, 0, 1], shape=(len(x), len(t))).toarray()