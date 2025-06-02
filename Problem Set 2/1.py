import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import constants

sigma = constants.Stefan_Boltzmann
alpha = 0.3
epsilon = 0.77
e_LW = 1 - epsilon / 2.0
S0 = 1367.0
rho = 1000.0
cp = 4218.0
h_ml = 70.0
Ceff = rho * cp * h_ml

def flux(T, S):
    return (1 - alpha) * S / 4 - e_LW * sigma * T ** 4

def newtown(f, x0, tol=1e-8, max_iter=100, n_tol=1e-7):
    for i in range(max_iter):
        fx = f(x0)
        dfx = (f(x0 + n_tol) - f(x0)) / n_tol
        x1 = x0 - fx / dfx
        if abs(x1 - x0) < tol and abs(fx) < tol:
            break
        x0 = x1
    return x1

def dT(t, T):
    return flux(T, S0) / Ceff * 24 * 3600

def simulate(T_initial, t_span):
    sol = solve_ivp(dT, t_span, [273.15], t_eval=np.linspace(0, 50 * 365, 51), method='Radau')
    sol.y[0] -= 273.15
    plt.plot(sol.t, sol.y[0])
    plt.xlabel('days')
    plt.ylabel('°C')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # (a)
    T0 = 20 + 273.15
    S = S0
    T = newtown(lambda T: flux(T, S), T0)
    print(f"Equilibrium temperature: {T - 273.15:.2f} °C")

    # (b) 50 years simulation
    t_span = (0, 50 * 365)
    simulate(T0, t_span)

    # (c) different values of S
    S_values = np.linspace(10, 200, 20)
    T_values = []
    for S_coeff in S_values:
        S = S_coeff * S0 / 100
        T = newtown(lambda T: flux(T, S), T0)
        T_values.append(T - 273.15)
    T_values = np.array(T_values)
    plt.plot(S_values, T_values)
    plt.xlabel('%S0')
    plt.ylabel('Equilibrium temperature (°C)')
    plt.grid()
    plt.show()

