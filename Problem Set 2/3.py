import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 1000
beta = 1.0
gamma = 1/3
S0 = 995
I0 = 5
R0 = 0

t_max = 60
dt = 0.1

t_euler = np.arange(0, t_max + dt, dt)
S_euler = np.zeros(len(t_euler))
I_euler = np.zeros(len(t_euler))
R_euler = np.zeros(len(t_euler))

S_euler[0] = S0
I_euler[0] = I0
R_euler[0] = R0

for n in range(len(t_euler) - 1):
    S = S_euler[n]
    I = I_euler[n]
    R = R_euler[n]

    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I

    S_euler[n + 1] = S + dS_dt * dt
    I_euler[n + 1] = I + dI_dt * dt
    R_euler[n + 1] = R + dR_dt * dt

plt.figure(figsize=(10, 6))
plt.plot(t_euler, S_euler, label='S')
plt.plot(t_euler, I_euler, label='I')
plt.plot(t_euler, R_euler, label='R')
plt.xlabel('Days)')
plt.ylabel('Number of Individuals')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()

def sir_system(t, y):
    S, I, R = y
    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

t_eval = np.linspace(0, t_max, 600)
solution = solve_ivp(sir_system, [0, t_max], [S0, I0, R0], t_eval=t_eval, method='RK45')
S_rk45, I_rk45, R_rk45 = solution.y
t_rk45 = solution.t

plt.figure(figsize=(10, 6))
plt.plot(t_rk45, S_rk45, label='S (RK45)')
plt.plot(t_rk45, I_rk45, label='I (RK45)')
plt.plot(t_rk45, R_rk45, label='R (RK45)')
plt.xlabel('Days)')
plt.ylabel('Number of Individuals')
plt.title('SIR Model (RK45)')
plt.legend()
plt.grid(True)
plt.show()
