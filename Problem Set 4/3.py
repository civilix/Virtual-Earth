import numpy as np
import matplotlib.pyplot as plt

L = 1.0
N = 50
k = 0.01
tf = 20.0

dx = L / N
x = np.linspace(0, L, N, endpoint=False)

dt = 0.4 * dx**2 / k
steps = int(tf / dt)

A = np.diag(np.full(N, -2.0)) + np.diag(np.full(N-1, 1.0), 1) + np.diag(np.full(N-1, 1.0), -1)
A[0, -1] = 1.0
A[-1, 0] = 1.0
A = A * (k / dx**2)

T = np.where(x <= L/2, 2*x/L, 2*(L-x)/L)
T_avg = np.mean(T)

plt.plot(x, T, label='t=0.0')

plot_steps = [int(0.2/dt), int(1.0/dt), int(5.0/dt), steps]

for i in range(1, steps + 1):
    T = T + dt * (A @ T)
    if i in plot_steps:
        t_now = i * dt
        plt.plot(x, T, label=f't={t_now:.1f}')

plt.axhline(T_avg, color='k', linestyle='--', label='Steady State')
plt.xlabel("x")
plt.ylabel("T")
plt.legend()
plt.show()