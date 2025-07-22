import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Lx = 1.0e6
Ly = 1.0e6
nx = 100
ny = 100

H = 4000.0
g = 9.81
beta = 2.3e-11

dx = Lx / nx
dy = Ly / ny

x_eta = np.linspace(dx/2, Lx - dx/2, nx)
y_eta = np.linspace(dy/2, Ly - dy/2, ny)
X_eta, Y_eta = np.meshgrid(x_eta, y_eta)

x_u = np.linspace(0, Lx, nx + 1)
y_u = np.linspace(dy/2, Ly - dy/2, ny)
_, Y_u = np.meshgrid(x_u, y_u)

x_v = np.linspace(dx/2, Lx - dx/2, nx)
y_v = np.linspace(0, Ly, ny + 1)
_, Y_v = np.meshgrid(x_v, y_v)

c_wave = np.sqrt(g * H)
dt = 0.5 * min(dx, dy) / c_wave
total_time = 12 * 3600
steps = int(total_time / dt)

eta = np.zeros((ny, nx))
u = np.zeros((ny, nx + 1))
v = np.zeros((ny + 1, nx))

amp = 1.0
eta += amp * np.exp(-((X_eta - Lx/2)**2 / (Lx/10)**2 + (Y_eta - Ly/2)**2 / (Ly/10)**2))

f_u = beta * (Y_u - Ly/2)
f_v = beta * (Y_v - Ly/2)

frames_data = []

for n in range(steps):
    if n % 50 == 0:
        u_c = 0.5 * (u[:, :-1] + u[:, 1:])
        v_c = 0.5 * (v[:-1, :] + v[1:, :])
        frames_data.append((eta.copy(), u_c.copy(), v_c.copy(), n * dt))

    u_new = u.copy()
    v_new = v.copy()
    eta_new = eta.copy()

    for j in range(ny):
        for i in range(nx):
            dudx = (u[j, i+1] - u[j, i]) / dx
            dvdy = (v[j+1, i] - v[j, i]) / dy
            eta_new[j, i] = eta[j, i] - H * dt * (dudx + dvdy)

    for j in range(ny):
        for i in range(1, nx):
            grad_eta_x = g * (eta[j, i] - eta[j, i-1]) / dx
            v_avg = 0.25 * (v[j, i-1] + v[j, i] + v[j+1, i-1] + v[j+1, i])
            coriolis_u = f_u[j, i] * v_avg
            u_new[j, i] = u[j, i] + dt * (coriolis_u - grad_eta_x)

    for j in range(1, ny):
        for i in range(nx):
            grad_eta_y = g * (eta[j, i] - eta[j-1, i]) / dy
            u_avg = 0.25 * (u[j-1, i] + u[j-1, i+1] + u[j, i] + u[j, i+1])
            coriolis_v = f_v[j, i] * u_avg
            v_new[j, i] = v[j, i] + dt * (-coriolis_v - grad_eta_y)

    eta, u, v = eta_new, u_new, v_new

    u[:, 0] = 0
    u[:, -1] = 0
    v[0, :] = 0
    v[-1, :] = 0

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

def update_plot(k):
    eta_d, u_d, v_d, t = frames_data[k]
    ax1.clear()
    ax2.clear()

    vmax = amp / 2
    ax1.set_title(f"Surface Displacement (η), t={t/3600:.1f} hrs")
    im = ax1.pcolormesh(X_eta/1e3, Y_eta/1e3, eta_d, cmap='seismic', vmin=-vmax, vmax=vmax)
    ax1.set_xlabel("x (km)")
    ax1.set_ylabel("y (km)")
    ax1.set_aspect('equal')

    skip = 5
    ax2.set_title("Velocity Field (u, v)")
    ax2.quiver(X_eta[::skip, ::skip]/1e3, Y_eta[::skip, ::skip]/1e3,
               u_d[::skip, ::skip], v_d[::skip, ::skip], scale=10)
    ax2.set_xlabel("x (km)")
    ax2.set_ylabel("y (km)")
    ax2.set_aspect('equal')
    ax2.set_xlim(0, Lx/1e3)
    ax2.set_ylim(0, Ly/1e3)

    fig.tight_layout()
    return fig,

ani = animation.FuncAnimation(fig, update_plot, frames=len(frames_data), interval=50)
ani.save('beta_plane_tsunami.gif', writer='pillow')