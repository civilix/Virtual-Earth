import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 3a
def lorenz(t, u, s, r, b):
    x, y, z = u
    dx = s * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return [dx, dy, dz]

s = 10
b = 8/3
u0 = [1.0, 1.0, 1.0]
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 10000)

rhos = [0.5, 10, 28]
fig = plt.figure(figsize=(18, 5))

for i, r in enumerate(rhos):
    ax = fig.add_subplot(1, 3, i + 1, projection='3d')
    sol = solve_ivp(lorenz, t_span, u0, args=(s, r, b), dense_output=True, t_eval=t_eval)
    ax.plot(sol.y[0], sol.y[1], sol.y[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"rho = {r}")

plt.show()

# 3b
r = 28
u1 = [0.0, 1.0, 0.0]
u2 = [0.0 + 1e-8, 1.0, 0.0]
t_span = [0, 50]
t_eval = np.linspace(t_span[0], t_span[1], 5000)

sol1 = solve_ivp(lorenz, t_span, u1, args=(s, r, b), t_eval=t_eval)
sol2 = solve_ivp(lorenz, t_span, u2, args=(s, r, b), t_eval=t_eval)

plt.figure(figsize=(12, 6))
plt.plot(sol1.t, sol1.y[0], label='x(t) for u0 = (0, 1, 0)')
plt.plot(sol2.t, sol2.y[0], label='x(t) for u0 = (1e-8, 1, 0)', linestyle='--')
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.show()

# 3c
s = 10
b = 8/3
r_vals = np.arange(0, 30.1, 0.2)

x_fp = []

for r in r_vals:
    fp_at_r = [0]
    if r > 1:
        x_pos = np.sqrt(b * (r - 1))
        fp_at_r.append(x_pos)
        fp_at_r.append(-x_pos)
    else:
        fp_at_r.append(np.nan)
        fp_at_r.append(np.nan)
    x_fp.append(fp_at_r)

x_fp = np.array(x_fp)

plt.figure(figsize=(10, 6))
plt.plot(r_vals, x_fp[:, 0], 'b')
plt.plot(r_vals, x_fp[:, 1], 'r')
plt.plot(r_vals, x_fp[:, 2], 'r')
plt.xlabel("rho")
plt.ylabel("x*")
plt.title("Fixed points of x as a function of rho")
plt.grid(True)
plt.show()

# 3d
s = 10
r = 28
b = 8/3

fp0 = np.array([0, 0, 0])

x_p = np.sqrt(b * (r - 1))
y_p = np.sqrt(b * (r - 1))
z_p = r - 1
fp_p = np.array([x_p, y_p, z_p])

x_n = -np.sqrt(b * (r - 1))
y_n = -np.sqrt(b * (r - 1))
z_n = r - 1
fp_n = np.array([x_n, y_n, z_n])

def jacobian(u, s, r, b):
    x, y, z = u
    return np.array([
        [-s, s, 0],
        [r - z, -1, -x],
        [y, x, -b]
    ])

j0 = jacobian(fp0, s, r, b)
jp = jacobian(fp_p, s, r, b)
jn = jacobian(fp_n, s, r, b)

eig0 = np.linalg.eigvals(j0)
eig_p = np.linalg.eigvals(jp)
eig_n = np.linalg.eigvals(jn)

print("Fixed Points:")
print(f"FP1: {fp0}")
print(f"FP2: {fp_p.round(4)}")
print(f"FP3: {fp_n.round(4)}")
print("\nEigenvalues:")
print(f"FP1: {np.round(eig0, 4)}")
print(f"FP2: {np.round(eig_p, 4)}")
print(f"FP3: {np.round(eig_n, 4)}")