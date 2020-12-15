import numpy as np
import matplotlib.pyplot as plt

def g(x):
    return np.sin(np.pi * x)


L = 1
T = 1
dx = 0.001
dt = 0.5 * dx**2
x = np.arange(0, L, dx)
t = np.arange(0, T, dt)

alpha = dt / dx**2

u = np.zeros(len(x))   # solution array
u_1 = np.zeros(len(x))
u_1[:] = g(x[:])
Nx = len(x) - 1

selected_times = [t[0], t[int(len(t)/2)], t[-1]]
stored_solution = np.zeros([len(selected_times), len(x)])

stored = 0
for time in t:
    u[1:Nx] = u_1[1:Nx] + alpha * (u_1[0:Nx - 1] - 2 * u_1[1:Nx] + u_1[2:Nx + 1])

    if time in selected_times:
        stored_solution[stored] = u
        stored += 1

    u[0] = 0
    u[Nx] = 0

    u_1, u = u, u_1

def analytic(x, t):
    return (np.exp(-(np.pi**2) * t)* np.sin(np.pi * x))


plt.plot(x, stored_solution[0], label="Num")
plt.plot(x, analytic(x, 0), label="analytic")
plt.legend()
plt.show()


plt.plot(x, stored_solution[1], label="Num")
plt.plot(x, analytic(x, 0.5), label="analytic")
plt.legend()
plt.show()


plt.plot(x, stored_solution[2], label="Num")
plt.plot(x, analytic(x, t[-1]), label="analytic")
plt.legend()
plt.show()