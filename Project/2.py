import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Domain parameters
Lx, Ly = 1.0, 1.0
Nx, Ny = 50, 50
dx, dy = Lx/(Nx-1), Ly/(Ny-1)

# Physical parameters
D = 0.01  # diffusion coefficient
vx, vy = 3.0, 0.0  # advection velocities

# 1) Build diffusion (Laplacian) operator with Neumann BC

# x-direction 1D Laplacian with zero-flux at boundaries
Ax = sp.lil_matrix((Nx, Nx))
for i in range(1, Nx-1):
    Ax[i, i-1] = 1.0
    Ax[i, i]   = -2.0
    Ax[i, i+1] = 1.0
# Left boundary (i=0): (c1 - c0) * 2 / dx^2
Ax[0, 0] = -2.0
Ax[0, 1] =  2.0
# Right boundary (i=Nx-1)
Ax[-1, -2] = 2.0
Ax[-1, -1] = -2.0
Ax = Ax.tocsr() / dx**2

# y-direction 1D Laplacian
Ay = sp.lil_matrix((Ny, Ny))
for j in range(1, Ny-1):
    Ay[j, j-1] = 1.0
    Ay[j, j]   = -2.0
    Ay[j, j+1] = 1.0
# Bottom boundary (j=0)
Ay[0, 0] = -2.0
Ay[0, 1] =  2.0
# Top boundary (j=Ny-1)
Ay[-1, -2] = 2.0
Ay[-1, -1] = -2.0
Ay = Ay.tocsr() / dy**2

Ix = sp.eye(Nx, format='csr')
Iy = sp.eye(Ny, format='csr')
Laplacian = sp.kron(Iy, Ax) + sp.kron(Ay, Ix)

# 2) Build upwind advection operator (upwind_1d already enforces zero advective flux at boundaries)
def upwind_1d(N, d, v):
    if v > 0:
        data, offsets = [np.ones(N-1), -np.ones(N-1)], [0, -1]
    else:
        data, offsets = [-np.ones(N-1), np.ones(N-1)], [1, 0]
    A_int = sp.diags(data, offsets, shape=(N-1, N-1), format='csr') / d
    A = sp.lil_matrix((N, N))
    A[1:, 1:] = A_int
    return A.tocsr()

Dx = upwind_1d(Nx, dx, vx)
Dy = upwind_1d(Ny, dy, vy)
Adv_x = sp.kron(Iy, Dx) * vx
Adv_y = sp.kron(Dy, Ix) * vy

# Combined operator
Operator = -Adv_x - Adv_y + D * Laplacian

# 3) Initial condition
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
sigma = 0.05
c = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2 * sigma**2)).flatten()

# 4) Time-stepping parameters
dt_adv = min(dx, dy) / (max(abs(vx), abs(vy)) + 1e-6) * 0.5
dt_diff = dx * dy / (4 * D) * 0.5
dt = min(dt_adv, dt_diff)
n_steps = 200

# 5) Precompute frames
frames = []
for _ in range(n_steps):
    c = c + dt * Operator.dot(c)
    frames.append(c.reshape((Ny, Nx)))

# 6) Plot & animate
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(frames[0], origin='lower', extent=(0, Lx, 0, Ly))
ax.set_title('Time step 0')
ax.set_xlabel('x'); ax.set_ylabel('y')

def update(i):
    im.set_data(frames[i])
    ax.set_title(f'Time step {i}')
    return [im]

anim = FuncAnimation(fig, update, frames=range(0, n_steps, 5), blit=True)
anim.save('advection_diffusion_neumann.gif', writer=PillowWriter(fps=30))
