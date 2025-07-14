import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 50, 50
dx, dy = Lx/(Nx-1), Ly/(Ny-1)

# Physical parameters
D = 0.01  # diffusion coefficient
vx, vy = 1.0, 0.0  # initial advection velocities

# Helper for 1D diffusion with Neumann BC (no-flux)
def diffusion_1d_neumann(N, d):
    main = -2.0 * np.ones(N)
    main[0] = -1.0
    main[-1] = -1.0
    off = np.ones(N-1)
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], format='csr') / d**2
    return A

# Helper for 1D upwind advection with reflective BC
def upwind_1d_reflective(N, d, v):
    if abs(v) < 1e-8:
        return sp.csr_matrix((N, N))
    # interior points only
    Ni = N - 2
    if v > 0:
        data = [np.ones(Ni), -np.ones(Ni)]
        offsets = [0, -1]
    else:
        data = [-np.ones(Ni), np.ones(Ni)]
        offsets = [1, 0]
    A_int = sp.diags(data, offsets, shape=(Ni, Ni), format='csr') / d
    # pad boundaries with zero rows/cols
    A = sp.lil_matrix((N, N))
    A[1:-1, 1:-1] = A_int
    return A.tocsr() * v

# Construct 2D operators
Ix = sp.eye(Nx, format='csr')
Iy = sp.eye(Ny, format='csr')
Ax = diffusion_1d_neumann(Nx, dx)
Ay = diffusion_1d_neumann(Ny, dy)
Laplacian = sp.kron(Iy, Ax, format='csr') + sp.kron(Ay, Ix, format='csr')

Dx = upwind_1d_reflective(Nx, dx, vx)
Dy = upwind_1d_reflective(Ny, dy, vy)
Adv_x = sp.kron(Iy, Dx, format='csr')
Adv_y = sp.kron(Dy, Ix, format='csr')

# Combined operator
Operator = -Adv_x - Adv_y + D * Laplacian

# Initial condition: Gaussian blob at center
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
sigma = 0.05
c = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2 * sigma**2)).flatten()

# Time-stepping parameters
dt_adv = min(dx, dy) / (max(abs(vx), abs(vy)) + 1e-6) * 0.4
dt_diff = dx * dy / (4 * D) * 0.4
dt = min(dt_adv, dt_diff)
n_steps = 200

# Set up figure
fig, ax = plt.subplots()
img = ax.imshow(c.reshape((Ny, Nx)), origin='lower', extent=(0, Lx, 0, Ly))
ax.set_title('t = 0.00')
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar = fig.colorbar(img, ax=ax, label='Concentration')

# Animation update function
def update(frame):
    global c
    c = c + dt * Operator.dot(c)
    C = c.reshape((Ny, Nx))
    img.set_data(C)
    ax.set_title(f't = {frame*dt:.2f}')
    return img,

ani = FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True)

# Display animation
ani.save('advection_diffusion_animation.gif', writer='ffmpeg', fps=30)