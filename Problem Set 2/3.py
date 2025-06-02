import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters for the SIR model
N = 1000           # Total population
beta = 1.0         # Contact rate per day
gamma = 1/3        # Recovery rate per day (1/γ = 3 days)
S0 = 995           # Initial susceptible
I0 = 5             # Initial infected
R0 = 0             # Initial recovered

# Time parameters
t_max = 60         # Simulate for 60 days
dt = 0.1           # Time step for forward Euler

# Time grid for forward Euler
t_euler = np.arange(0, t_max + dt, dt)

# Arrays to store S, I, R over time for forward Euler
S_euler = np.zeros(len(t_euler))
I_euler = np.zeros(len(t_euler))
R_euler = np.zeros(len(t_euler))

# Initial conditions
S_euler[0] = S0
I_euler[0] = I0
R_euler[0] = R0

# Forward Euler integration
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

# Plotting S, I, R using forward Euler
plt.figure(figsize=(10, 6))
plt.plot(t_euler, S_euler, label='Susceptible (Euler)')
plt.plot(t_euler, I_euler, label='Infected (Euler)')
plt.plot(t_euler, R_euler, label='Recovered (Euler)')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SIR Model (Forward Euler)')
plt.legend()
plt.grid(True)
plt.show()

# Define the SIR system for use with solve_ivp (RK45)
def sir_system(t, y):
    S, I, R = y
    dS_dt = -beta * S * I / N
    dI_dt = beta * S * I / N - gamma * I
    dR_dt = gamma * I
    return [dS_dt, dI_dt, dR_dt]

# Time grid for solve_ivp
t_eval = np.linspace(0, t_max, 600)

# Solve with RK45 (similar to Matlab's ode45)
solution = solve_ivp(sir_system, [0, t_max], [S0, I0, R0], t_eval=t_eval, method='RK45')

# Extract the solution
S_rk45, I_rk45, R_rk45 = solution.y
t_rk45 = solution.t

# Plotting S, I, R using RK45
plt.figure(figsize=(10, 6))
plt.plot(t_rk45, S_rk45, label='Susceptible (RK45)')
plt.plot(t_rk45, I_rk45, label='Infected (RK45)')
plt.plot(t_rk45, R_rk45, label='Recovered (RK45)')
plt.xlabel('Time (days)')
plt.ylabel('Number of Individuals')
plt.title('SIR Model (RK45)')
plt.legend()
plt.grid(True)
plt.show()
