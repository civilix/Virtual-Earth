import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
import matplotlib.pyplot as plt

So = 1367.0
eps = 0.77
eps_LW = 1 - eps / 2.0
sigma = constants.Stefan_Boltzmann
rho = 1000.0
cp = 4218.0
h_ml = 70.0
Ceff = rho * cp * h_ml
a1 = 0.3
a2 = 0.7
TU = 10.0  # C
TL = -10.0  # C

def albedo(T):
    """Temperature-dependent albedo (T in K)."""
    T_C = T - 273.15
    if T_C > TU:
        return a1
    elif T_C < TL:
        return a2
    else:
        return a2 + (a1 - a2) * (T_C - TL) / (TU - TL)

# Time step: 1 day
def dTdt(t, T, S):
    """dT/dt in K/day given temperature T (K) and solar input S (W/m^2)."""
    net_flux = (1 - albedo(T)) * S / 4 - eps_LW * sigma * T**4
    return (net_flux / Ceff) * 86400

t_final = 365 * 100

# 1)
ratios_down = np.arange(2.0, 0.39, -0.1)
Ts_equil_down = []
T0 = 273.15

for r in ratios_down:
    S = r * So
    sol = solve_ivp(dTdt, [0, t_final], [T0], args=(S,), atol=1e-6, rtol=1e-6)
    T_eq = sol.y[0, -1]
    Ts_equil_down.append(T_eq)
    T0 = T_eq

# 2)
ratios_up = np.arange(0.5, 2.01, 0.1)
Ts_equil_up = []

for r in ratios_up:
    S = r * So
    sol = solve_ivp(dTdt, [0, t_final], [T0], args=(S,), atol=1e-6, rtol=1e-6)
    T_eq = sol.y[0, -1]
    Ts_equil_up.append(T_eq)
    T0 = T_eq

plt.figure()
plt.plot(ratios_down, Ts_equil_down, label='Decreasing S')
plt.plot(ratios_up, Ts_equil_up, label='Increasing S')
plt.xlabel('S / S₀')
plt.ylabel('Equilibrium Temperature (K)')
plt.title('Equilibrium Temperature vs Solar Constant Ratio')
plt.legend()
plt.show()