import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

def DiffusionMatrix(n):
    diagonals = [
        -2 * np.ones(n),
        np.ones(n),
        np.ones(n)
    ]
    offsets = [0, -1, 1]
    A = diags(diagonals, offsets, shape=(n, n), format='csr')
    return A

L = 1.0
kappa = 0.001
dx = 0.05
T_left = 0.0
T_right = 10.0
t_end = 200

n_nodes = int(L/dx) + 1
x_full = np.linspace(0, L, n_nodes)
N = n_nodes - 2
x = x_full[1:-1]

M = kappa / dx**2 * DiffusionMatrix(N)
b = np.zeros(N)
b[-1] = kappa / dx**2 * T_right

def simulate(dt, method='forward'):
    r = kappa * dt / dx**2
    times = np.arange(0, t_end+1, 20)
    T = np.zeros(N)
    T[x > L/2] = T_right
    sol = {0: T.copy()}
    steps = int(np.ceil(t_end / dt))

    if method == 'forward':
        for n in range(1, steps+1):
            T = T + dt * (M.dot(T) + b)
            t = n * dt
            if np.isclose(t % 20, 0, atol=dt/2):
                sol[int(round(t))] = T.copy()
        title = f"Forward Euler (dt={dt}s, r={r:.2f})"

    elif method == 'backward':
        A_imp = eye(N, format='csc') - dt * M
        LU = splu(A_imp)
        for n in range(1, steps+1):
            rhs = T + dt * b
            T = LU.solve(rhs)
            t = n * dt
            if np.isclose(t % 20, 0, atol=dt/2):
                sol[int(round(t))] = T.copy()
        title = f"Backward Euler (dt={dt}s, r={r:.2f})"

    plt.figure()
    for tt in times:
        T_full = np.concatenate(([T_left], sol[tt], [T_right]))
        plt.plot(x_full, T_full, label=f"t={tt}s")
    plt.xlabel("x [m]")
    plt.ylabel("T [°C]")
    plt.title(title)
    plt.legend()

if __name__ == '__main__':
    simulate(dt=1.0, method='forward')
    simulate(dt=2.0, method='forward')
    simulate(dt=2.0, method='backward')
    plt.show()
