import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 66436000
beta = 0.5
epsilon = 1 / 5
gamma = 1 / 6
mu = 731213 / 66436000 / 365

def seir_ode(t, y):
    S, E, I, R = y
    dS_dt = -beta * S * I / N + mu * N - mu * S
    dE_dt = beta * S * I / N - epsilon * E - mu * E
    dI_dt = epsilon * E - gamma * I - mu * I
    dR_dt = gamma * I - mu * R
    return [dS_dt, dE_dt, dI_dt, dR_dt]

y0 = [N - 1, 0, 1, 0]
t_span = (0, 160)
t_eval = np.linspace(0, 160, 1601)

solution = solve_ivp(seir_ode, t_span, y0, t_eval=t_eval, vectorized=True)

S = solution.y[0] / N
E = solution.y[1] / N
I = solution.y[2] / N
R = solution.y[3] / N
t = solution.t

plt.figure(figsize=(10, 6))
plt.plot(t, S, label='S')
plt.plot(t, E, label='E')
plt.plot(t, I, label='I')
plt.plot(t, R, label='R')
plt.xlabel('Days)')
plt.ylabel('Proportion of population')
plt.title('SEIR Model')
plt.legend()
plt.grid(True)
plt.show()
