import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def ftcs(u0, nx, nt, dx, dt, c):
    u = u0.copy()
    un = np.zeros(nx)
    alpha = c * dt / dx
    for _ in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = un[i] - alpha / 2 * (un[i+1] - un[i-1])
        u[0] = un[0] - alpha / 2 * (un[1] - un[nx-2])
        u[nx-1] = u[0]
    return u

def upwind(u0, nx, nt, dx, dt, c):
    u = u0.copy()
    un = np.zeros(nx)
    alpha = c * dt / dx
    for _ in range(nt):
        un = u.copy()
        for i in range(1, nx):
            u[i] = un[i] - alpha * (un[i] - un[i-1])
        u[0] = un[0] - alpha * (un[0] - un[nx-2])
    return u

def crank_nicolson(u0, nx, nt, dx, dt, c):
    u = u0.copy()
    beta = c * dt / (4 * dx)
    A = np.diag(np.ones(nx))
    A += np.diag(np.ones(nx-1) * beta, 1)
    A += np.diag(np.ones(nx-1) * -beta, -1)
    A[0, -1] = -beta
    A[-1, 0] = beta

    B = np.diag(np.ones(nx))
    B += np.diag(np.ones(nx-1) * -beta, 1)
    B += np.diag(np.ones(nx-1) * beta, -1)
    B[0, -1] = beta
    B[-1, 0] = -beta

    for _ in range(nt):
        b = B @ u
        u = solve(A, b)
    return u

def btcs(u0, nx, nt, dx, dt, c):
    u = u0.copy()
    gamma = c * dt / (2 * dx)
    A = np.diag(np.ones(nx))
    A += np.diag(np.ones(nx-1) * gamma, 1)
    A += np.diag(np.ones(nx-1) * -gamma, -1)
    A[0, -1] = -gamma
    A[-1, 0] = gamma

    for _ in range(nt):
        u = solve(A, u)
    return u

def lax_friedrichs(u0, nx, nt, dx, dt, c):
    u = u0.copy()
    un = np.zeros(nx)
    alpha = c * dt / dx
    for _ in range(nt):
        un = u.copy()
        for i in range(1, nx - 1):
            u[i] = 0.5 * (un[i+1] + un[i-1]) - alpha / 2 * (un[i+1] - un[i-1])
        u[0] = 0.5 * (un[1] + un[nx-2]) - alpha / 2 * (un[1] - un[nx-2])
        u[nx-1] = u[0]
    return u

L = 1.0
c = 0.1
nx = 101
dx = L / (nx - 1)
t_final = 10.0
alpha_cfl = 0.5
dt = alpha_cfl * dx / c
nt = int(t_final / dt)

x = np.linspace(0, L, nx)

u0_hat = np.zeros(nx)
u0_hat[int(0.4/dx):int(0.6/dx)+1] = 1.0

u0_gauss = np.exp(-100 * (x - 0.5)**2)

analytical_hat = np.zeros(nx)
analytical_hat[int(0.4/dx):int(0.6/dx)+1] = 1.0

x_analytical_gauss = (x - c * t_final) % L
analytical_gauss = np.exp(-100 * (x_analytical_gauss - 0.5)**2)

schemes = {
    "FTCS": ftcs,
    "Upwind": upwind,
    "Crank-Nicolson": crank_nicolson,
    "BTCS": btcs,
    "Lax-Friedrichs": lax_friedrichs
}

initial_conditions = {
    "Top Hat": (u0_hat, analytical_hat),
    "Gaussian": (u0_gauss, analytical_gauss)
}

for ic_name, (u0, u_analytical) in initial_conditions.items():
    plt.figure(figsize=(12, 8))
    plt.plot(x, u0, 'k--', label="Initial Condition")

    for name, func in schemes.items():
        u_final = func(u0, nx, nt, dx, dt, c)
        plt.plot(x, u_final, label=name)

    plt.plot(x, u_analytical, 'k-', lw=2, label="Analytical Solution")
    plt.title(f"1D Advection with {ic_name} Initial Condition (c={c}, t={t_final})")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.ylim(-0.2, 1.2)
    plt.tight_layout()
    plt.show()