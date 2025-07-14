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

# Build diffusion (Laplacian) operator
main_x = -2.0 * np.ones(Nx)
off_x = np.ones(Nx-1)
Ax = sp.diags([off_x, main_x, off_x], [-1, 0, 1], format='csr') / dx**2
main_y = -2.0 * np.ones(Ny)
off_y = np.ones(Ny-1)
Ay = sp.diags([off_y, main_y, off_y], [-1, 0, 1], format='csr') / dy**2
Ix = sp.eye(Nx, format='csr')
Iy = sp.eye(Ny, format='csr')
Laplacian = sp.kron(Iy, Ax) + sp.kron(Ay, Ix)

# Build 1D upwind advection matrix
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

# Initial condition
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)
sigma = 0.05
c = np.exp(-((X-0.5)**2 + (Y-0.5)**2) / (2 * sigma**2)).flatten()

# Time-stepping parameters
dt_adv = min(dx, dy) / (max(abs(vx), abs(vy)) + 1e-6) * 0.5
dt_diff = dx * dy / (4 * D) * 0.5
dt = min(dt_adv, dt_diff)
n_steps = 200

# Precompute all frames
frames = []
for _ in range(n_steps):
    c = c + dt * Operator.dot(c)
    frames.append(c.reshape((Ny, Nx)))

# Setup figure
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(frames[0], origin='lower', extent=(0, Lx, 0, Ly))
ax.set_title('Time step 0')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Animation update function
def update(frame_idx):
    im.set_data(frames[frame_idx])
    ax.set_title(f'Time step {frame_idx}')
    return [im]

# Create FuncAnimation
anim = FuncAnimation(fig, update, frames=range(0, n_steps, 5), blit=True)

# Save as GIF
anim.save('advection_diffusion_animation.gif', writer='ffmpeg', fps=30)
